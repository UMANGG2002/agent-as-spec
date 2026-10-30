"""Time-ordered unique ids (UUIDv7 layout: 48-bit ms timestamp + random).

Sortable by creation time, so B-tree inserts stay append-mostly (no page
splits from random UUIDs at high write rates) and ids double as a cursor
for keyset pagination.
"""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76  # version 7
    value |= ((rand >> 68) & 0xFFF) << 64  # 12 bits rand_a
    value |= 0b10 << 62  # RFC 4122 variant
    value |= rand & ((1 << 62) - 1)
    return uuid.UUID(int=value)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid7().hex}"
