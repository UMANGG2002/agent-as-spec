"""Per-tenant BYOK credential vault. Minimal build: a Postgres/SQLite table
with application-level Fernet envelope encryption, exactly as the architecture
doc's data-model note describes (swap the Fernet key for a KMS in production).

Nothing outside this module ever sees a decrypted secret except at the moment
a tool wrapper needs it to make an outbound call.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Literal

from cryptography.fernet import Fernet

from app.config import settings


def _derive_fernet_key(material: str) -> bytes:
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_fernet = Fernet(_derive_fernet_key(settings.jwt_signing_key))

VaultEntryKind = Literal["oauth_app", "api_key", "refresh_token"]


@dataclass
class VaultEntry:
    id: str
    tenant_id: str
    kind: VaultEntryKind
    ciphertext: bytes

    def reveal(self) -> str:
        return _fernet.decrypt(self.ciphertext).decode("utf-8")


class InMemoryVault:
    """Minimal build uses an in-process store seeded at startup / via the API.
    Swap for a `CredentialVaultEntry` table (see marketplace/models.py) backed
    read/write path without changing any caller."""

    def __init__(self) -> None:
        self._entries: dict[str, VaultEntry] = {}

    def put(self, entry_id: str, tenant_id: str, kind: VaultEntryKind, secret: str) -> VaultEntry:
        entry = VaultEntry(
            id=entry_id,
            tenant_id=tenant_id,
            kind=kind,
            ciphertext=_fernet.encrypt(secret.encode("utf-8")),
        )
        self._entries[entry_id] = entry
        return entry

    def get(self, entry_id: str, *, tenant_id: str) -> VaultEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None or entry.tenant_id != tenant_id:
            return None
        return entry


vault = InMemoryVault()
