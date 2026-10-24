"""Find personal data (PII) in text, entirely on this machine.

Two kinds of detector:
- regex + validators for structured IDs (email, phone, PAN, Aadhaar, card, IFSC, UPI,
  bank account, date of birth). Checksums (Verhoeff for Aadhaar, Luhn for cards) and
  context words ("A/c", "DOB") keep false positives down.
- a local spaCy NER model for names, organisations and places. Best-effort only.

The tripwire uses ONLY the regex detectors: they are precise, so they can safely block a request.
"""

import re
import threading
from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class Finding:
    type: str
    value: str
    start: int
    end: int


# ---------- validators ----------
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]  # fmt: skip
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]  # fmt: skip


def verhoeff_valid(digits: str) -> bool:
    """Aadhaar numbers end in a Verhoeff check digit: random 12-digit numbers fail 9 times in 10."""
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


def luhn_valid(digits: str) -> bool:
    """Card numbers end in a Luhn check digit."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


# ---------- regex detectors (order = priority when two overlap) ----------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
UPI_RE = re.compile(r"\b[A-Za-z0-9._-]{2,}@[A-Za-z]{3,}\b(?!\.[A-Za-z])")  # name@okaxis: no dot after @
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
ACCOUNT_RE = re.compile(r"(?i)\b(?:a/c|acct|account)(?:\s*(?:no|number|num))?\.?\s*[:#-]?\s*(\d[\d ]{7,20}\d)")
PHONE_RE = re.compile(r"(?<![\d+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b")  # 4th letter = holder type
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
DATE = r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4})"
DOB_RE = re.compile(rf"(?i)\b(?:dob|d\.o\.b\.?|date of birth|born on)\s*[:-]?\s*({DATE})")


def _regex_findings(text: str) -> list[Finding]:
    found: list[Finding] = []
    for m in EMAIL_RE.finditer(text):
        found.append(Finding("EMAIL", m.group(), m.start(), m.end()))
    for m in UPI_RE.finditer(text):
        found.append(Finding("UPI", m.group(), m.start(), m.end()))
    for m in CARD_RE.finditer(text):
        d = _digits(m.group())
        if 13 <= len(d) <= 19 and luhn_valid(d):
            found.append(Finding("CARD", m.group(), m.start(), m.end()))
    for m in AADHAAR_RE.finditer(text):
        if verhoeff_valid(_digits(m.group())):
            found.append(Finding("AADHAAR", m.group(), m.start(), m.end()))
    for m in ACCOUNT_RE.finditer(text):
        found.append(Finding("ACCOUNT", m.group(1), m.start(1), m.end(1)))
    for m in PHONE_RE.finditer(text):
        found.append(Finding("PHONE", m.group(), m.start(), m.end()))
    for m in PAN_RE.finditer(text):
        found.append(Finding("PAN", m.group(), m.start(), m.end()))
    for m in IFSC_RE.finditer(text):
        found.append(Finding("IFSC", m.group(), m.start(), m.end()))
    for m in DOB_RE.finditer(text):
        found.append(Finding("DOB", m.group(1), m.start(1), m.end(1)))
    return found


REGEX_TYPES = ("EMAIL", "UPI", "CARD", "AADHAAR", "ACCOUNT", "PHONE", "PAN", "IFSC", "DOB")


# ---------- NER (names, organisations, places) ----------
NER_LABELS = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION"}
_nlp = None
_nlp_lock = threading.Lock()


def _get_nlp():
    global _nlp
    with _nlp_lock:
        if _nlp is None:
            import spacy

            _nlp = spacy.load(get_settings().ner_model, disable=["lemmatizer"])
    return _nlp


# Words a small NER model often mistakes for names in contracts and forms (measured on the sample
# agreement: "Tenant", "Landlord", "Aadhaar" came back as PERSON/ORG). Hiding them hurts answers.
NOT_NAMES = {
    "tenant", "landlord", "lessor", "lessee", "licensor", "licensee", "owner", "buyer", "seller", "employer",
    "employee", "party", "parties", "witness", "company", "agreement", "borrower", "lender", "guarantor",
    "applicant", "customer", "client", "aadhaar", "rent", "deposit", "society", "property",
}  # fmt: skip


def _looks_like_a_name(value: str) -> bool:
    if len(value) < 3 or not any(c.isupper() for c in value) or re.search(r"\d{4,}", value):
        return False
    if value.isupper():  # headings ("MAINTENANCE") and acronyms ("PAN", "DOB") are not names
        return False
    return value.casefold().removeprefix("the ") not in NOT_NAMES


def _ner_findings(text: str) -> list[Finding]:
    found = []
    nlp = _get_nlp()
    for start in range(0, len(text), 100_000):  # spaCy's default max length is 1,000,000 chars
        piece = text[start : start + 100_000]
        for ent in nlp(piece).ents:
            kind = NER_LABELS.get(ent.label_)
            value = ent.text.strip()
            if kind and _looks_like_a_name(value):
                found.append(Finding(kind, value, start + ent.start_char, start + ent.end_char))
    return found


def _drop_overlaps(findings: list[Finding]) -> list[Finding]:
    priority = {t: i for i, t in enumerate(REGEX_TYPES + ("PERSON", "ORG", "LOCATION"))}
    findings = sorted(findings, key=lambda f: (f.start, priority[f.type], -(f.end - f.start)))
    kept: list[Finding] = []
    for f in findings:
        if kept and f.start < kept[-1].end:
            if priority[f.type] < priority[kept[-1].type]:
                kept[-1] = f
            continue
        kept.append(f)
    return kept


def detect(text: str, use_ner: bool | None = None) -> list[Finding]:
    """All PII findings in text, non-overlapping, in order of position."""
    use_ner = get_settings().ner_enabled if use_ner is None else use_ner
    findings = _regex_findings(text)
    if use_ner:
        findings += _ner_findings(text)
    return _drop_overlaps(findings)


def detect_regex(text: str) -> list[Finding]:
    """Precise detectors only (what the tripwire uses)."""
    return _drop_overlaps(_regex_findings(text))
