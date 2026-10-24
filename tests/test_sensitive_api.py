"""Sensitive mode end to end: no planted PII may reach the (fake) LLM; the user still sees real values."""

import json
import re

import pytest

from app.db import get_pool
from app.llm.prompts import ANSWER_SYSTEM, MAP_SYSTEM
from tests.test_privacy import AGREEMENT, PLANTED

pytestmark = [pytest.mark.db, pytest.mark.slow]
ALL_VALUES = [v for vs in PLANTED.values() for v in vs]


def fake_llm_reply(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == MAP_SYSTEM:
        ids = re.findall(r'<section id="(\d+)"', user)
        people = sorted(set(re.findall(r"\[PERSON_\d+\]", user)))
        return json.dumps(
            {"sections": [{"id": int(i), "summary": f"Section {i} mentions {', '.join(people[:2])}.",
                           "entities": [{"name": p, "kind": "person"} for p in people]} for i in ids]}
        )  # fmt: skip
    if "deduplicate" in system:
        return '{"groups": []}'
    if "overviews" in system:
        return "A rental agreement between [PERSON_1] and [PERSON_2]."
    assert system == ANSWER_SYSTEM
    loc = re.findall(r'loc="([^"]+)"', user)[0]
    phone = re.findall(r"\[PHONE_\d+\]", user)
    return f"The tenant's phone is {phone[0] if phone else 'unknown'} [document.txt, {loc}]."


@pytest.fixture
def agreement(api, fake_llm):
    fake_llm.handler = fake_llm_reply
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = 'Ananya_Iyer_lease.txt'")
    response = api.post(
        "/documents",
        files={"file": ("Ananya_Iyer_lease.txt", AGREEMENT.encode(), "text/plain")},
        data={"sensitive": "true"},
    )
    assert response.status_code == 202, response.text
    doc = response.json()
    yield doc
    api.delete(f"/documents/{doc['id']}")


def sent_text(fake_llm) -> str:
    return "\n".join(m["content"] for call in fake_llm.calls for m in call)


def test_nothing_sensitive_reaches_the_llm(api, agreement, fake_llm):
    doc = api.get(f"/documents/{agreement['id']}").json()
    assert doc["status"] == "ready" and doc["sensitive"] and doc["map_status"] == "ready"
    answer = api.post(f"/documents/{agreement['id']}/ask", json={"question": "What is Ananya Iyer's phone number?"})
    assert answer.status_code == 200, answer.text
    sent = sent_text(fake_llm)
    assert fake_llm.calls, "the LLM should have been called (map + answer)"
    leaked = [v for v in ALL_VALUES if v in sent]
    assert leaked == [], f"leaked to LLM: {leaked}"
    assert "Ananya_Iyer_lease.txt" not in sent  # the filename contains a name too
    assert "[PERSON_" in sent  # the question's name became a placeholder


def test_user_sees_real_values_and_can_see_redacted(api, agreement):
    body = api.post(f"/documents/{agreement['id']}/ask", json={"question": "What is the tenant's phone number?"}).json()
    assert body["sensitive"] is True
    assert "[PHONE_" in body["answer_redacted"]
    assert "[PHONE_" not in body["answer"] and any(p in body["answer"] for p in PLANTED["PHONE"])
    assert body["llm_payload"][0]["content"] == ANSWER_SYSTEM  # transparency: the exact prompt is returned
    assert all(s["source"] == "Ananya_Iyer_lease.txt" for s in body["sources"])  # real name shown locally


def test_document_map_is_restored_for_the_owner(api, agreement):
    restored = api.get(f"/documents/{agreement['id']}/map").json()
    redacted = api.get(f"/documents/{agreement['id']}/map", params={"redacted": "true"}).json()
    assert "[PERSON_" in redacted["summary"]
    assert "[PERSON_" not in restored["summary"] and "Rohan Mehta" in restored["summary"]


def test_pii_report_counts_types_without_values(api, agreement):
    report = api.get(f"/documents/{agreement['id']}/pii-report").json()
    by_type = {t["type"]: t for t in report["types"]}
    for pii_type in ("EMAIL", "PHONE", "PAN", "AADHAAR", "CARD", "IFSC", "ACCOUNT", "UPI", "DOB"):
        assert by_type[pii_type]["distinct"] == len(PLANTED[pii_type]), pii_type
    assert by_type["PHONE"]["occurrences"] == 3  # landlord's number appears twice
    assert not any(v in json.dumps(report) for v in ALL_VALUES)


def test_deleting_removes_the_vault(api, fake_llm):
    fake_llm.handler = fake_llm_reply
    doc = api.post(
        "/documents", files={"file": ("lease2.txt", (AGREEMENT + "\nAnnex.").encode())}, data={"sensitive": "true"}
    ).json()
    with get_pool().connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM pii_vault WHERE document_id = %s", [doc["id"]]).fetchone()["n"]
    assert api.delete(f"/documents/{doc['id']}").status_code == 204
    with get_pool().connection() as conn:
        assert (
            conn.execute("SELECT count(*) AS n FROM pii_vault WHERE document_id = %s", [doc["id"]]).fetchone()["n"] == 0
        )


def test_tripwire_blocks_when_redaction_misses_something(api, agreement, fake_llm, monkeypatch):
    # simulate a redaction bug: put a raw PAN back into a stored chunk
    with get_pool().connection() as conn:
        conn.execute(
            "UPDATE chunks SET text = text || %s WHERE document_id = %s", [f" PAN {PLANTED['PAN'][0]}", agreement["id"]]
        )
    calls_before = len(fake_llm.calls)
    response = api.post(f"/documents/{agreement['id']}/ask", json={"question": "What is the monthly rent?"})
    assert response.status_code == 503 and "Blocked" in response.json()["detail"]
    assert len(fake_llm.calls) == calls_before  # nothing was sent


def test_same_file_cannot_switch_modes_silently(api, agreement):
    again = api.post("/documents", files={"file": ("again.txt", AGREEMENT.encode())}, data={"sensitive": "false"})
    assert again.status_code == 409
