#!/usr/bin/env python3
"""Fetch the exact upstream FlyBody source assets used for port-equivalence diagnostics.

The files are cached under artifacts/ and are never committed.  We fetch the source
repository tarball at a pinned commit so diagnostics compare virtual-fly against an
immutable upstream model rather than against a moving branch.
"""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
import urllib.request


UPSTREAM_COMMIT = "d015e9bfe441bd90ae431bac24c55cb74bdbce26"
UPSTREAM_TARBALL = f"https://github.com/TuragaLab/flybody/archive/{UPSTREAM_COMMIT}.tar.gz"
OUTPUT_ROOT = Path("artifacts/upstream") / f"flybody-{UPSTREAM_COMMIT}"
ASSET_RELATIVE = Path("flybody/fruitfly/assets")
EXPECTED_XML = OUTPUT_ROOT / ASSET_RELATIVE / "fruitfly.xml"


def main() -> int:
    if EXPECTED_XML.exists():
        print(f"upstream_flybody_source={EXPECTED_XML}")
        print("upstream_flybody_source=READY")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"downloading={UPSTREAM_TARBALL}")
    with urllib.request.urlopen(UPSTREAM_TARBALL, timeout=120) as response:
        payload = response.read()

    prefix = f"flybody-{UPSTREAM_COMMIT}/{ASSET_RELATIVE.as_posix()}/"
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = Path(member.name[len(prefix) :])
            if not relative.parts:
                continue
            destination = OUTPUT_ROOT / ASSET_RELATIVE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(member)
            if src is None:
                continue
            destination.write_bytes(src.read())
            extracted += 1

    if not EXPECTED_XML.exists():
        raise RuntimeError(
            f"upstream source extraction did not produce {EXPECTED_XML}; extracted={extracted}"
        )
    print(f"extracted_files={extracted}")
    print(f"upstream_flybody_source={EXPECTED_XML}")
    print("upstream_flybody_source=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
