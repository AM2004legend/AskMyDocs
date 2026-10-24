"""Sensitive mode: detectors, placeholders, vault, tripwire. Synthetic PII only (samples/sensitive/)."""

import json
import logging
from pathlib import Path

import pytest

from app.llm.client import LLMBlockedPII, LLMClient, Provider
from app.privacy.detectors import detect, detect_regex, luhn_valid, verhoeff_valid
from app.privacy.redactor import Redactor, restore
from app.privacy.tripwire import scan_payload, sensitive_scope

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sensitive"
PLANTED: dict[str, list[str]] = json.loads((SAMPLE / "planted_pii.json").read_text(encoding="utf-8"))
AGREEMENT = (SAMPLE / "rental_agreement_synthetic.txt").read_text(encoding="utf-8")
REGEX_PLANTED = {t: v for t, v in PLANTED.items() if t not in {"PERSON", "ORG"}}


def test_checksums():
    assert all(verhoeff_valid(a.replace(" ", "")) for a in PLANTED["AADHAAR"])
    assert not verhoeff_valid("234567890123")
    assert luhn_valid("4111111111111111") and not luhn_valid("4111111111111112")


@pytest.mark.parametrize(("pii_type", "value"), [(t, v) for t, vs in REGEX_PLANTED.items() for v in vs])
def test_every_planted_value_is_detected(pii_type, value):
    found = {(f.type, f.value) for f in detect_regex(AGREEMENT)}
    assert (pii_type, value) in found


def test_no_false_alarms_on_ordinary_text():
    ordinary = (
        "The rent is Rs. 28,000 payable by the 5th. Invoice 2026-03-01, order 123456789012 (not an Aadhaar), "
        "card 4111 1111 1111 1112 fails Luhn, chapter 14, page 300, and email-like text user at domain."
    )
    assert detect_regex(ordinary) == []


def test_upi_is_not_mistaken_for_email():
    types = {f.type for f in detect_regex("pay ananya.iyer@okhdfcbank or mail ananya@example.org")}
    assert types == {"UPI", "EMAIL"}


@pytest.mark.slow
def test_ner_finds_names_and_organisations():
    found = {f.value for f in detect(AGREEMENT, use_ner=True) if f.type in {"PERSON", "ORG"}}
    # best-effort: report which planted names NER finds (the README states the measured recall)
    assert {"Rohan Mehta", "Ananya Iyer"} <= found


def test_placeholders_are_consistent_and_restorable():
    r = Redactor()
    a = r.redact("Call +91 98220 41736 or mail x.y@example.com.", use_ner=False)
    b = r.redact("Phone again: 9822041736. Second mail: z@example.com", use_ner=False)
    assert a == "Call [PHONE_1] or mail [EMAIL_1]."
    assert b == "Phone again: [PHONE_1]. Second mail: [EMAIL_2]"  # same number, other spelling -> same placeholder
    values = {p: v for p, (_, v) in r.values.items()}
    assert (
        restore("Reach PERSON_X at [PHONE_1] / EMAIL_2", values) == "Reach PERSON_X at +91 98220 41736 / z@example.com"
    )


def test_known_values_are_replaced_everywhere():
    r = Redactor([("PERSON_1", "PERSON", "Ananya Iyer")], new_prefix="Q")
    assert r.redact("What does Ananya Iyer pay? Mail me at me@example.com", use_ner=False) == (
        "What does [PERSON_1] pay? Mail me at [EMAIL_Q1]"
    )


def test_whole_agreement_redacts_every_planted_value():
    r = Redactor()
    r.learn(AGREEMENT, use_ner=False)
    for name in PLANTED["PERSON"] + PLANTED["ORG"]:  # as if NER had found them (tested separately above)
        r.placeholder_for(type("F", (), {"type": "PERSON", "value": name})())
    redacted = r.redact(AGREEMENT, use_ner=False)
    leaked = [v for vs in PLANTED.values() for v in vs if v in redacted]
    assert leaked == []
    assert restore(redacted, {p: v for p, (_, v) in r.values.items()}) == AGREEMENT  # nothing lost


KNOWN = {v: t for t, vs in PLANTED.items() for v in vs}  # what the vault holds for this document


def test_tripwire_catches_100_percent_of_planted_pii_even_without_context():
    """PRD acceptance: 100% of planted regex-type PII. Bare values (no "A/c", no "DOB") included."""
    missed = []
    with sensitive_scope(True, known_values=KNOWN):
        for pii_type, values in PLANTED.items():
            for value in values:
                payload = [{"role": "user", "content": f"<excerpt>Some contract text ... {value} ... more</excerpt>"}]
                if pii_type not in scan_payload(payload):
                    missed.append(pii_type)
    assert missed == []


def test_tripwire_catches_other_spellings_of_known_numbers():
    with sensitive_scope(True, known_values={"+91 98220 41736": "PHONE"}):
        assert scan_payload([{"content": "call 9822041736"}]) == ["PHONE"]
        assert scan_payload([{"content": "call [PHONE_1]"}]) == []


def test_regex_detectors_alone_need_context_for_account_and_dob():
    """Honest limit, measured: without the vault check these two types need their keywords."""
    bare = {t: scan_payload([{"content": f"x {PLANTED[t][0]} y"}]) for t in ("ACCOUNT", "DOB")}
    assert bare == {"ACCOUNT": [], "DOB": []}
    assert scan_payload([{"content": f"A/c No. {PLANTED['ACCOUNT'][0]}, DOB: {PLANTED['DOB'][0]}"}]) == [
        "ACCOUNT",
        "DOB",
    ]


def test_tripwire_blocks_request_and_logs_only_types(caplog):
    calls = []
    client = LLMClient(
        [Provider("p", "u", "k", "m")],
        make_client=lambda p: type("C", (), {"chat": type("X", (), {"completions": type("Y", (), {
            "create": staticmethod(lambda **kw: calls.append(kw))})})})(),
    )  # fmt: skip
    leaked = PLANTED["PAN"][0]
    with caplog.at_level(logging.WARNING), sensitive_scope(True), pytest.raises(LLMBlockedPII, match="PAN"):
        client.chat([{"role": "user", "content": f"The tenant PAN is {leaked}"}])
    assert calls == []  # nothing was sent
    assert "PAN" in caplog.text and leaked not in caplog.text  # type logged, value never


def test_tripwire_is_off_for_normal_documents():
    calls = []

    class Fake:
        class chat:  # noqa: N801 - mimics openai client shape
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})})]})

    client = LLMClient([Provider("p", "u", "k", "m")], make_client=lambda p: Fake)
    assert client.chat([{"role": "user", "content": "mail me at someone@example.com"}]) == "ok"
    assert len(calls) == 1


def test_redacted_payload_passes_tripwire():
    r = Redactor()
    r.learn(AGREEMENT, use_ner=False)
    assert scan_payload([{"role": "user", "content": r.redact(AGREEMENT, use_ner=False)}]) == []


def test_vault_encrypts_values():
    from cryptography.fernet import Fernet

    from app.privacy import vault

    token = vault._fernet().encrypt(b"ABCPM4821K").decode()
    assert "ABCPM4821K" not in token
    assert vault._fernet().decrypt(token.encode()) == b"ABCPM4821K"
    assert len(Fernet.generate_key()) == 44 and len(vault.generate_key()) == 44
