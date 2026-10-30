"""JWT signing key ring with rotation.

  - local/test: HS256 with `jwt_signing_key` (kid "dev").
  - staging/production: RS256/ES256. `JWT_PRIVATE_KEYS_JSON` maps kid ->
    PEM private key; `JWT_ACTIVE_KID` picks the one that signs new tokens.
    `JWT_PUBLIC_KEYS_JSON` can add public keys of retired kids so tokens
    they signed still verify until they expire.

Rotation without downtime: add the new key (both keys verify), switch
`JWT_ACTIVE_KID`, wait one max token TTL, then remove the old key.
Public keys are served at `/.well-known/jwks.json`, so other services (and
the runner workers) verify tokens without ever holding a signing key.

Every token carries `kid`, `iss`, `aud`, `iat`, `exp`, `jti`; verification
pins the algorithm list (no `alg=none` / HS-vs-RS confusion), requires
those claims, and allows a few seconds of clock skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from app.config import settings


@dataclass
class KeyRing:
    algorithm: str
    active_kid: str
    signing_key: Any
    verify_keys: dict[str, Any]

    def jwks(self) -> dict:
        if self.algorithm.startswith("HS"):
            return {"keys": []}  # never publish a symmetric key
        keys = []
        for kid, key in self.verify_keys.items():
            alg = RSAAlgorithm if self.algorithm.startswith("RS") else ECAlgorithm
            jwk = json.loads(alg.to_jwk(key))
            jwk.update({"kid": kid, "use": "sig", "alg": self.algorithm})
            keys.append(jwk)
        return {"keys": keys}


@lru_cache(maxsize=1)
def keyring() -> KeyRing:
    alg = settings.jwt_algorithm.upper()
    if alg.startswith("HS"):
        return KeyRing(alg, "dev", settings.jwt_signing_key, {"dev": settings.jwt_signing_key})

    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

    private = json.loads(settings.jwt_private_keys_json or "{}")
    if not private or settings.jwt_active_kid not in private:
        raise RuntimeError("JWT_PRIVATE_KEYS_JSON must contain JWT_ACTIVE_KID")
    loaded = {kid: load_pem_private_key(pem.encode(), password=None) for kid, pem in private.items()}
    verify = {kid: key.public_key() for kid, key in loaded.items()}
    for kid, pem in json.loads(settings.jwt_public_keys_json or "{}").items():
        verify.setdefault(kid, load_pem_public_key(pem.encode()))
    return KeyRing(alg, settings.jwt_active_kid, loaded[settings.jwt_active_kid], verify)


def sign(payload: dict) -> str:
    ring = keyring()
    return jwt.encode(payload, ring.signing_key, algorithm=ring.algorithm, headers={"kid": ring.active_kid})


def decode(token: str) -> dict:
    ring = keyring()
    kid = jwt.get_unverified_header(token).get("kid", "dev")
    key = ring.verify_keys.get(kid)
    if key is None:
        raise jwt.InvalidTokenError(f"unknown signing key '{kid}'")
    return jwt.decode(
        token,
        key,
        algorithms=[ring.algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        leeway=5,
        options={"require": ["exp", "iat", "iss", "aud", "jti"]},
    )


def reset_for_tests() -> None:
    keyring.cache_clear()
