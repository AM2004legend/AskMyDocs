"""Replace PII with consistent placeholders, and put the real values back afterwards.

"Rahul Sharma" -> [PERSON_1] every time it appears in the document, so the LLM can still
reason ("[PERSON_1] signed with [ORG_1]") without ever seeing the real name.

Values are matched on a canonical form, so "+91 98765 43210" and "9876543210" share one
placeholder. After detection, every known value is also replaced wherever else it appears
verbatim, which catches names NER missed in other sentences.
"""

import re
from collections import Counter

from app.privacy.detectors import REGEX_TYPES, Finding, detect

TYPES = REGEX_TYPES + ("PERSON", "ORG", "LOCATION")
# the LLM sometimes drops the brackets, so match both "[PERSON_1]" and "PERSON_1"
PLACEHOLDER_RE = re.compile(r"\[?\b(" + "|".join(TYPES) + r")_(Q?\d+)\b\]?")


NAME_TYPES = {"PERSON", "ORG", "LOCATION"}


def _key(pii_type: str, value: str) -> tuple[str, str]:
    # NER may call the same text PERSON in one sentence and ORG in another: one placeholder for both
    return ("NAME" if pii_type in NAME_TYPES else pii_type, canonical(pii_type, value))


def canonical(pii_type: str, value: str) -> str:
    if pii_type in {"EMAIL", "UPI"}:
        return value.strip().lower()
    if pii_type in {"PHONE"}:
        return re.sub(r"\D", "", value)[-10:]
    if pii_type in {"CARD", "AADHAAR", "ACCOUNT"}:
        return re.sub(r"\D", "", value)
    if pii_type in {"PAN", "IFSC"}:
        return value.strip().upper()
    return re.sub(r"\s+", " ", value).strip().casefold()


class Redactor:
    """One per document (or per question, seeded from the document's vault)."""

    def __init__(self, entries: list[tuple[str, str, str]] | None = None, new_prefix: str = ""):
        """entries: existing (placeholder, type, value) from the vault. New placeholders get `new_prefix`."""
        self.new_prefix = new_prefix
        self.by_key: dict[tuple[str, str], str] = {}
        self.values: dict[str, tuple[str, str]] = {}  # placeholder -> (type, value)
        self.surfaces: dict[str, str] = {}  # exact text seen -> placeholder
        self.counters: Counter = Counter()
        for placeholder, pii_type, value in entries or []:
            self.by_key[_key(pii_type, value)] = placeholder
            self.values[placeholder] = (pii_type, value)
            self.surfaces[value] = placeholder
            if not new_prefix:  # continue numbering after existing placeholders
                number = re.search(r"(\d+)$", placeholder)
                self.counters[pii_type] = max(self.counters[pii_type], int(number.group(1)) if number else 0)

    def placeholder_for(self, f: Finding) -> str:
        key = _key(f.type, f.value)
        if key not in self.by_key:
            self.counters[f.type] += 1
            placeholder = f"{f.type}_{self.new_prefix}{self.counters[f.type]}"
            self.by_key[key] = placeholder
            self.values[placeholder] = (f.type, f.value)
        placeholder = self.by_key[key]
        self.surfaces.setdefault(f.value, placeholder)
        return placeholder

    def learn(self, text: str, use_ner: bool | None = None) -> None:
        """Detect PII and assign placeholders without changing text (first pass over a whole document)."""
        for f in detect(text, use_ner):
            self.placeholder_for(f)

    def redact(self, text: str, use_ner: bool | None = None) -> str:
        findings = detect(text, use_ner)
        out = text
        for f in reversed(findings):  # right to left, so earlier offsets stay valid
            out = out[: f.start] + f"[{self.placeholder_for(f)}]" + out[f.end :]
        return self.replace_known(out)

    def replace_known(self, text: str) -> str:
        """Replace every value already in the map wherever it appears verbatim."""
        surfaces = sorted((s for s in self.surfaces if len(s) >= 3), key=len, reverse=True)
        if not surfaces:
            return text
        # case-insensitive, like the tripwire's check: "ROHAN MEHTA" in a heading is still Rohan Mehta
        lookup = {s.casefold(): p for s, p in self.surfaces.items()}
        pattern = re.compile(r"(?<![\w\[])(" + "|".join(re.escape(s) for s in surfaces) + r")(?!\w)", re.IGNORECASE)
        return pattern.sub(lambda m: f"[{lookup[m.group(1).casefold()]}]", text)

    def entries(self) -> list[tuple[str, str, str]]:
        return [(placeholder, t, v) for placeholder, (t, v) in self.values.items()]


def restore(text: str, values: dict[str, str]) -> str:
    """Put real values back. values: placeholder -> real value. Unknown placeholders stay as they are."""

    def swap(m: re.Match) -> str:
        return values.get(f"{m.group(1)}_{m.group(2)}", m.group(0))

    return PLACEHOLDER_RE.sub(swap, text)
