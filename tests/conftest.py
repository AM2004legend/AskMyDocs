"""Shared test setup.

- Tests use their own database (askmydocs_test), created automatically, never your real data.
- LLM keys are blanked so no test can ever call Gemini/Groq; tests that need an LLM use FakeLLM,
  a real LLMClient whose network call is faked (so the tripwire is always exercised).
- Tests marked `db` are skipped when Postgres isn't running.
"""

import os
from types import SimpleNamespace

import psycopg
import pytest

ADMIN_URL = os.environ.get("TEST_DATABASE_ADMIN_URL", "postgresql://askmydocs:askmydocs@localhost:5433/askmydocs")
TEST_DB = "askmydocs_test"

# must happen before app.config.get_settings() is first called
os.environ["DATABASE_URL"] = ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB
os.environ["LLM_API_KEY"] = ""
os.environ["FALLBACK_LLM_API_KEY"] = ""
os.environ["LLM_MIN_INTERVAL_SECONDS"] = "0"
os.environ["PII_VAULT_KEY"] = "V2hW0eY7o2h9m1mHq3n8Yp6r4s5t7u9w1x3z5b7d9f0="  # throwaway key, tests only

from app.llm.client import LLMClient, Provider  # noqa: E402 - must come after the env setup above


def _prepare_test_db() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", [TEST_DB]).fetchone()
            if not exists:
                conn.execute(f"CREATE DATABASE {TEST_DB}")
        return True
    except psycopg.OperationalError:
        return False


DB_OK = _prepare_test_db()
if os.environ.get("CI") and not DB_OK:  # in CI a missing database is a failure, not a reason to skip
    raise RuntimeError(f"CI: cannot reach the test database at {ADMIN_URL}")


def pytest_collection_modifyitems(config, items):
    if DB_OK:
        return
    skip = pytest.mark.skip(reason="Postgres not reachable (run: docker compose up -d db)")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


class FakeLLM(LLMClient):
    """A REAL LLMClient with only the network call faked.

    So every test still runs the real tripwire, retry and fallback code. `handler(messages) -> str`
    decides the reply; every request that actually got "sent" is recorded in `calls`.
    """

    def __init__(self, handler=None):
        self.calls: list[list[dict]] = []
        self.handler = handler or (lambda messages: "I couldn't find that in this document.")
        fake = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                fake.calls.append(kwargs["messages"])
                content = fake.handler(kwargs["messages"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        transport = SimpleNamespace(chat=SimpleNamespace(completions=_Completions))
        super().__init__([Provider("fake", "http://fake", "fake", "fake-model")], max_retries=0,
                         make_client=lambda p: transport)  # fmt: skip


@pytest.fixture
def fake_llm():
    from app.llm import client

    fake = FakeLLM()
    old = client._client
    client._client = fake
    yield fake
    client._client = old


@pytest.fixture(scope="session")
def api():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def make_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid text PDF by hand (one Helvetica text line per input line)."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages))), len(pages)
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        lines = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in text.split("\n")]
        stream = "BT /F1 11 Tf 40 760 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
