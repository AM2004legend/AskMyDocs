"""The PII vault: placeholder -> real value, encrypted at rest with Fernet (AES + HMAC).

The key lives only in .env (PII_VAULT_KEY). Without it the stored values are unreadable,
which is the point - but it also means: lose the key, lose the ability to restore answers.
Rows are deleted with their document (ON DELETE CASCADE).
"""

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings
from app.db import get_pool


class VaultError(Exception):
    """Vault misconfigured or unreadable; the message is safe to show."""


def generate_key() -> str:
    return Fernet.generate_key().decode()


def _fernet() -> Fernet:
    key = get_settings().pii_vault_key.get_secret_value()
    if not key:
        raise VaultError(
            "Sensitive mode needs PII_VAULT_KEY in .env. Create one with: "
            'python -c "from app.privacy.vault import generate_key; print(generate_key())"'
        )
    try:
        return Fernet(key.encode())
    except ValueError as exc:
        raise VaultError("PII_VAULT_KEY in .env is not a valid Fernet key.") from exc


def is_configured() -> bool:
    try:
        _fernet()
        return True
    except VaultError:
        return False


def save_entries(conn, doc_id: int, entries: list[tuple[str, str, str]]) -> None:
    """entries: (placeholder, pii_type, value). Runs inside the caller's transaction."""
    fernet = _fernet()
    conn.execute("DELETE FROM pii_vault WHERE document_id = %s", [doc_id])
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO pii_vault (document_id, placeholder, pii_type, encrypted_value) VALUES (%s, %s, %s, %s)",
            [(doc_id, p, t, fernet.encrypt(v.encode()).decode()) for p, t, v in entries],
        )


def load_entries(doc_id: int) -> list[tuple[str, str, str]]:
    fernet = _fernet()
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT placeholder, pii_type, encrypted_value FROM pii_vault WHERE document_id = %s ORDER BY id",
            [doc_id],
        ).fetchall()
    try:
        return [(r["placeholder"], r["pii_type"], fernet.decrypt(r["encrypted_value"].encode()).decode()) for r in rows]
    except InvalidToken as exc:
        raise VaultError("The vault can't be decrypted: PII_VAULT_KEY has changed since upload.") from exc
