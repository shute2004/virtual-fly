#!/usr/bin/env python3
"""Fetch the official FlyBody baseline wing-beat pattern from Janelia Figshare.

The FlyBody source maps its ``flight-imitation-dataset`` bundle to Figshare file
51196859.  Only ``wing_pattern_fmech.npy`` is needed by virtual-fly, so this
prefetcher downloads the official archive and extracts that member into the
ignored artifacts cache rather than committing upstream binary data.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.request
import zipfile

import numpy as np


DEFAULT_URL = "https://janelia.figshare.com/ndownloader/files/51196859"
DEFAULT_OUTPUT = Path("artifacts/flybody-data/wing_pattern_fmech.npy")
TARGET_BASENAME = "wing_pattern_fmech.npy"


def validate_pattern(path: Path) -> tuple[int, int]:
    pattern = np.load(path, allow_pickle=False)
    if pattern.ndim != 2 or pattern.shape[1] != 3 or pattern.shape[0] < 20:
        raise RuntimeError(
            f"unexpected FlyBody wing pattern shape at {path}: {pattern.shape}; "
            "expected (N, 3) for yaw/roll/pitch"
        )
    if not np.all(np.isfinite(pattern)):
        raise RuntimeError(f"non-finite values in FlyBody wing pattern: {path}")
    return int(pattern.shape[0]), int(pattern.shape[1])


def main() -> int:
    url = os.environ.get("VF_FLYBODY_FLIGHT_DATA_URL", DEFAULT_URL)
    output = Path(os.environ.get("VF_FLYBODY_WING_PATTERN", str(DEFAULT_OUTPUT)))
    timeout_s = float(os.environ.get("VF_FLYBODY_FLIGHT_DATA_TIMEOUT_S", "180"))
    if timeout_s <= 0:
        raise ValueError("VF_FLYBODY_FLIGHT_DATA_TIMEOUT_S must be positive")

    if output.exists():
        rows, cols = validate_pattern(output)
        print(f"wing_pattern={output}")
        print(f"wing_pattern_shape=({rows},{cols})")
        print("flybody_flight_data=READY")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(timeout_s)

    with tempfile.TemporaryDirectory(prefix="virtual-fly-flight-data-") as tmp_dir:
        archive = Path(tmp_dir) / "flight-imitation-dataset.zip"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "virtual-fly/1.0 FlyBody data prefetch"},
        )
        print(f"downloading={url}", flush=True)
        with urllib.request.urlopen(request, timeout=timeout_s) as response, archive.open("wb") as dst:
            shutil.copyfileobj(response, dst, length=1024 * 1024)

        if not zipfile.is_zipfile(archive):
            raise RuntimeError(
                "official FlyBody flight-imitation download was not a ZIP archive; "
                f"source={url}"
            )
        with zipfile.ZipFile(archive) as zf:
            matches = [name for name in zf.namelist() if Path(name).name == TARGET_BASENAME]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one {TARGET_BASENAME} in official archive, found {matches}"
                )
            member = matches[0]
            staged = Path(tmp_dir) / TARGET_BASENAME
            with zf.open(member) as src, staged.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            rows, cols = validate_pattern(staged)
            staged.replace(output)

    print(f"wing_pattern={output}")
    print(f"wing_pattern_shape=({rows},{cols})")
    print("flybody_flight_data=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
