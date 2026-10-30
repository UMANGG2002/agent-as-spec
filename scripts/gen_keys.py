"""Generate production key material (print as env lines):

    python scripts/gen_keys.py >> .env.production

  - JWT_PRIVATE_KEYS_JSON / JWT_ACTIVE_KID: a fresh RS256 signing key.
    To rotate: generate a new kid, add it to the JSON alongside the old one,
    switch JWT_ACTIVE_KID, remove the old key after one token TTL.
  - VAULT_KEYS_JSON: a fresh Fernet data key (prepend new keys to rotate).
"""

from __future__ import annotations

import json
import time

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

kid = time.strftime("k%Y%m%d")
key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                        serialization.NoEncryption()).decode()
print(f"JWT_ALGORITHM=RS256")
print(f"JWT_ACTIVE_KID={kid}")
print(f"JWT_PRIVATE_KEYS_JSON='{json.dumps({kid: pem})}'")
print(f"VAULT_KEYS_JSON='{json.dumps([Fernet.generate_key().decode()])}'")
