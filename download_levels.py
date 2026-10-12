"""
download_levels.py — fetch a real Elasto Mania level pack from the Elasto Online API.

The `.lev` files are copyrighted level assets (they are NOT open source - only the game
*engine* was released). They are therefore deliberately NOT committed to this repository;
this script downloads them into `levels/<pack>/` on demand so the repo stays clean of
redistributed game assets.

Usage:
    python3 download_levels.py --pack Best          # 20 levels, "The Best Elasto Mania Levels"
    python3 download_levels.py --pack Best --list   # just show what the pack contains

API discovery: elmaonline.net is a React SPA; its client (assets/index-*.js) calls
    https://api.elma.online/api/levelpack/<name>?levels=1   -> pack + level list
    https://api.elma.online/dl/level/<LevelIndex>           -> the .lev file itself
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

API = "https://api.elma.online"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "elma-rl-research",
                                               "Accept": "application/json, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_pack(name: str) -> dict:
    return json.loads(_get(f"{API}/api/levelpack/{name}?levels=1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="Best", help="level pack name (e.g. Best, Int, ...)")
    ap.add_argument("--out", default="levels", help="output directory")
    ap.add_argument("--list", action="store_true", help="only list the pack contents")
    a = ap.parse_args()

    pack = fetch_pack(a.pack)
    levels = pack.get("levels", [])
    print(f"pack {pack['LevelPackName']!r}: {pack['LevelPackLongName']!r} "
          f"by {pack['KuskiData']['Kuski']} -- {len(levels)} levels")
    for lv in levels:
        print(f"  {lv['LevelIndex']:>8d}  {lv['LevelName']}")
    if a.list:
        return

    outdir = os.path.join(a.out, a.pack)
    os.makedirs(outdir, exist_ok=True)
    ok = 0
    for lv in levels:
        path = os.path.join(outdir, f"{lv['LevelName']}_{lv['LevelIndex']}.lev")
        if os.path.exists(path):
            print(f"  have  {path}")
            ok += 1
            continue
        try:
            data = _get(f"{API}/dl/level/{lv['LevelIndex']}")
            with open(path, "wb") as f:
                f.write(data)
            print(f"  got   {path}  ({len(data)} bytes)")
            ok += 1
        except Exception as e:                                  # noqa: BLE001
            print(f"  FAIL  {lv['LevelName']}: {e}")
    print(f"\n{ok}/{len(levels)} levels in {outdir}")
    print("NOTE: these files are copyrighted game assets - use them locally for research, "
          "do not redistribute them.")


if __name__ == "__main__":
    sys.exit(main())
