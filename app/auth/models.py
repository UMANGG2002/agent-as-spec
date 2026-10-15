"""Identity + delegation data shapes. See the architecture doc's
"Multi-tenancy, identity & OAuth delegation chain" section — this module is
that section made concrete."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IdentityContext:
    """Minted once at the edge, threaded unchanged through planner, factory,
    tool wrapper, guardrail plugin, and trace span attributes."""

    tenant_id: str
    brand_id: str
    user_id: str
    scopes: tuple[str, ...]
    # The delegation chain so far, oldest first: platform -> tenant -> run -> tool-call.
    act_chain: tuple[str, ...] = field(default_factory=tuple)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or "*" in self.scopes

    def delegate(self, actor: str, *, narrowed_scopes: tuple[str, ...] | None = None) -> "IdentityContext":
        """Produce the next, narrower hop in the delegation chain. Never
        widens scope -- only intersects or keeps it."""
        scopes = self.scopes if narrowed_scopes is None else tuple(s for s in narrowed_scopes if s in self.scopes)
        return IdentityContext(
            tenant_id=self.tenant_id,
            brand_id=self.brand_id,
            user_id=self.user_id,
            scopes=scopes,
            act_chain=self.act_chain + (actor,),
        )


@dataclass(frozen=True)
class DelegatedToken:
    """An ephemeral, narrow-scoped token for exactly one hop. Not persisted
    long-term -- see the architecture doc's data model note on this."""

    jti: str
    identity: IdentityContext
    scope: str
    expires_at: float
    encoded: str

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @staticmethod
    def new_jti() -> str:
        return uuid.uuid4().hex
