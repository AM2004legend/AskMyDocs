"""Last line of defence: scan every outbound LLM request made for a sensitive document.

Code that works on a sensitive document runs inside `sensitive_scope(known_values=...)`. While
that scope is active, LLMClient.chat() calls `scan_payload()` before sending anything. Two checks:

1. the precise regex detectors (email, phone, PAN, Aadhaar, card, IFSC, UPI, ...), and
2. the document's OWN vault values: every value we hid at ingest is known, so it can be caught
   even where no pattern could recognise it (a bare account number, a date of birth without
   "DOB", a name only NER found).

If either finds something, the request is blocked and only the PII *types* are logged.
A context variable (not a global) keeps the scope per thread/request.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar

from app.privacy.detectors import detect_regex

NUMERIC_TYPES = {"PHONE", "CARD", "AADHAAR", "ACCOUNT"}

_active: ContextVar[bool] = ContextVar("sensitive_scope", default=False)
_known: ContextVar[tuple[re.Pattern, ...]] = ContextVar("sensitive_known_values", default=())
_known_types: ContextVar[tuple[str, ...]] = ContextVar("sensitive_known_types", default=())


def _pattern(value: str, pii_type: str) -> re.Pattern:
    if pii_type in NUMERIC_TYPES:
        digits = re.sub(r"\D", "", value)
        if pii_type == "PHONE":
            digits = digits[-10:]
        # same digits, any spacing: "98220 41736" == "9822041736"
        return re.compile(r"(?<!\d)" + r"[ -]?".join(digits) + r"(?!\d)")
    return re.compile(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", re.IGNORECASE)


def compile_known(known_values: dict[str, str]) -> tuple[tuple[re.Pattern, ...], tuple[str, ...]]:
    """known_values: real value -> PII type. Values shorter than 4 characters are skipped (too generic)."""
    items = [(v, t) for v, t in known_values.items() if len(v.strip()) >= 4]
    return tuple(_pattern(v, t) for v, t in items), tuple(t for _, t in items)


@contextmanager
def sensitive_scope(active: bool = True, known_values: dict[str, str] | None = None):
    patterns, types = compile_known(known_values or {}) if active else ((), ())
    tokens = (_active.set(active), _known.set(patterns), _known_types.set(types))
    try:
        yield
    finally:
        _known_types.reset(tokens[2])
        _known.reset(tokens[1])
        _active.reset(tokens[0])


def is_active() -> bool:
    return _active.get()


def scan_payload(messages: list[dict]) -> list[str]:
    """Sorted PII types found in the message contents (never the values)."""
    types = set()
    patterns, known_types = _known.get(), _known_types.get()
    for message in messages:
        content = str(message.get("content", ""))
        types.update(f.type for f in detect_regex(content))
        # our own fixed system prompt never holds document text; skipping it means an NER false
        # positive like "Rules" in the vault can't block every request
        if message.get("role") != "system":
            types.update(t for p, t in zip(patterns, known_types, strict=True) if p.search(content))
    return sorted(types)
