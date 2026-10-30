"""Per-tenant BYOK credential vault with versioned envelope encryption.

Secrets are encrypted with `MultiFernet` over a list of data keys
(`VAULT_KEYS_JSON`, newest first): encryption always uses the newest key,
decryption tries each, and `rotate()` re-encrypts a ciphertext under the
newest key. Rotating = prepend a key, re-encrypt in the background, then
drop the old one -- no downtime. In production the data keys come from a
KMS (wrap/unwrap at boot) rather than an env var; the interface is the same.

Locally a key is derived from the dev JWT secret so nothing needs setting.
Nothing outside this module ever sees a decrypted secret except at the
moment a tool wrapper needs it to make an outbound call; entries are
tenant-scoped on every read.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from cryptography.fernet import Fernet, MultiFernet

from app.config import settings


def _derive_fernet_key(material: str) -> bytes:
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


@lru_cache(maxsize=1)
def _cipher() -> MultiFernet:
    if settings.vault_keys_json:
        keys = [Fernet(k.encode() if isinstance(k, str) else k) for k in json.loads(settings.vault_keys_json)]
    else:
        keys = [Fernet(_derive_fernet_key(settings.jwt_signing_key))]
    return MultiFernet(keys)


def encrypt(secret: str) -> bytes:
    return _cipher().encrypt(secret.encode("utf-8"))


def decrypt(ciphertext: bytes) -> str:
    return _cipher().decrypt(ciphertext).decode("utf-8")


def rotate(ciphertext: bytes) -> bytes:
    return _cipher().rotate(ciphertext)


VaultEntryKind = Literal["oauth_app", "api_key", "refresh_token"]


@dataclass
class VaultEntry:
    id: str
    tenant_id: str
    kind: VaultEntryKind
    ciphertext: bytes

    def reveal(self) -> str:
        return decrypt(self.ciphertext)


class InMemoryVault:
    """Local/dev store. Production persists `VaultEntry` rows (ciphertext
    only) in Postgres under row-level security; the interface is the same."""

    def __init__(self) -> None:
        self._entries: dict[str, VaultEntry] = {}

    def put(self, entry_id: str, tenant_id: str, kind: VaultEntryKind, secret: str) -> VaultEntry:
        entry = VaultEntry(id=entry_id, tenant_id=tenant_id, kind=kind, ciphertext=encrypt(secret))
        self._entries[entry_id] = entry
        return entry

    def get(self, entry_id: str, *, tenant_id: str) -> VaultEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None or entry.tenant_id != tenant_id:
            return None
        return entry


vault = InMemoryVault()
