"""Save a fix-loop result (fix_*.json) as a shareable run for the studio: runs/<name>.json.

Strips machine-specific paths and trims bulky fields. No API keys are ever in these files
(agents only see a dummy key; the proxy holds the real one).

    .venv/bin/python scripts/save_run.py fix_incident.json incident_response_2026-10-28
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH_PATTERNS = [
    (re.compile(r"/(?:private/)?var/folders/[^\s\"']+?/(rg_agent_audit_[^/\s\"']+)"), r"<tmp>/\1"),
    (re.compile(r"/(?:Users|home)/[^/\s\"']+"), "~"),
]


def scrub(value):
    if isinstance(value, str):
        for pat, rep in PATH_PATTERNS:
            value = pat.sub(rep, value)
        return value
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def main() -> None:
    src, name = Path(sys.argv[1]), sys.argv[2]
    data = json.loads(src.read_text())
    for rnd in data.get("rounds", []):
        for c in rnd:
            if c.get("raw"):
                c["raw"] = c["raw"][:20000]
            c.pop("workdir", None)
    data["saved_as"] = name
    out = ROOT / "runs" / f"{name}.json"
    out.write_text(json.dumps(scrub(data), indent=1))
    leftover = re.findall(r"/Users/[^/\s\"']+", out.read_text())
    print(f"Saved {out} ({out.stat().st_size // 1024} KB); machine paths left: {len(leftover)}")


if __name__ == "__main__":
    main()
