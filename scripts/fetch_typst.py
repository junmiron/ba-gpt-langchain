#!/usr/bin/env python3
"""Download the Typst binary into vendor/typst so PDF export can use the primary engine.

Run:  python scripts/fetch_typst.py
Typst is discovered by `ba_agent.artifacts.typst.find_typst()` in this order:
  BA_TYPST_BIN -> `typst` on PATH -> vendor/typst/**/typst(.exe)
The vendored binary is gitignored; this script makes the setup reproducible.
"""

from __future__ import annotations

import argparse
import io
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = REPO_ROOT / "vendor" / "typst"
RELEASE = "https://github.com/typst/typst/releases/latest/download/"


def asset_name() -> str:
    system = sys.platform
    machine = platform.machine().lower()
    if system.startswith("win"):
        return "typst-x86_64-pc-windows-msvc.zip"
    if system == "darwin":
        return "typst-arm64-apple-darwin.zip" if machine in {"arm64", "aarch64"} else "typst-x86_64-apple-darwin.zip"
    return "typst-x86_64-unknown-linux-gnu.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None, help="Override the release asset name.")
    args = parser.parse_args()

    found = [path for path in VENDOR_DIR.rglob("typst*") if path.is_file() and path.suffix in {"", ".exe"}]
    if found:
        print(f"Typst already present: {found[0]}")
        return 0

    asset = args.asset or asset_name()
    url = RELEASE + asset
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(payload))
    entries = archive.namelist()
    archive.extractall(VENDOR_DIR)

    found = [path for path in VENDOR_DIR.rglob("typst*") if path.is_file() and path.suffix in {"", ".exe"}]
    print(f"Extracted {len(entries)} entries into {VENDOR_DIR}")
    for path in found:
        print(f"  {path}")
    if not found:
        print("No typst binary found after extraction — check the asset name.", file=sys.stderr)
        return 1
    print("Ready: the PDF renderer picks this up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
