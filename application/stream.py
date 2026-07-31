"""Bootstrap: load full ContinuousWalkStream from last known-good commit.

Temporary restore after accidental PLACEHOLDER overwrite.
"""
from __future__ import annotations

import urllib.request

_URL = (
    "https://raw.githubusercontent.com/17xbaphomet/vector-toon-pipeline/"
    "db22389/application/stream.py"
)
_ns: dict = {}
with urllib.request.urlopen(_URL, timeout=30) as resp:
    _src = resp.read().decode("utf-8")
exec(compile(_src, "stream_remote.py", "exec"), _ns)
globals().update({k: v for k, v in _ns.items() if not k.startswith("_")})
