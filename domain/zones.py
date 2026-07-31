"""Bootstrap: load full zones module from last known-good commit."""
from __future__ import annotations

import urllib.request

_URL = (
    "https://raw.githubusercontent.com/17xbaphomet/vector-toon-pipeline/"
    "db22389/domain/zones.py"
)
_ns: dict = {}
with urllib.request.urlopen(_URL, timeout=30) as resp:
    _src = resp.read().decode("utf-8")
exec(compile(_src, "zones_remote.py", "exec"), _ns)
globals().update({k: v for k, v in _ns.items() if not k.startswith("_")})
