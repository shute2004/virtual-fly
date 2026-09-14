#!/usr/bin/env python3
"""Robustly prefetch FlyGym's FlyBody meshes without redownloading valid files.

This intentionally delegates object listing, integrity checks, and downloads to
FlyGym itself. The only behavior added here is:

- use a persistent staging directory so valid files survive a failed attempt;
- apply a finite socket timeout so a stalled HTTP transfer does not block forever;
- retry the same FlyGym download routine, which then skips files already verified.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import sys
import time

from flygym.compose.fly.flybody import FLYBODY_FULLSIZE_MESH_DIR
from flygym.utils.assets_lazy_loading import (
    S3_ROOT_PREFIX,
    _download_prefix,
    get_cache_root,
)


def main() -> int:
    retries = int(os.environ.get("VF_FLYBODY_ASSET_RETRIES", "8"))
    timeout_s = float(os.environ.get("VF_FLYBODY_ASSET_TIMEOUT_S", "90"))
    if retries < 1:
        raise ValueError("VF_FLYBODY_ASSET_RETRIES must be >= 1")
    if timeout_s <= 0:
        raise ValueError("VF_FLYBODY_ASSET_TIMEOUT_S must be > 0")

    socket.setdefaulttimeout(timeout_s)

    rel_path = Path(FLYBODY_FULLSIZE_MESH_DIR)
    cache_dir = get_cache_root() / rel_path
    if cache_dir.is_dir():
        print(cache_dir)
        return 0

    staging = cache_dir.parent / f".{rel_path.name}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    prefix = f"{S3_ROOT_PREFIX}/{rel_path.as_posix()}"

    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_prefix(prefix, staging)
            last_error = None
            break
        except (OSError, TimeoutError) as exc:
            last_error = exc
            print(
                f"FlyBody asset fetch failed ({attempt}/{retries}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(min(attempt, 3))

    if last_error is not None:
        raise last_error

    if cache_dir.is_dir():
        # Another process may have completed the same cache while we downloaded.
        shutil.rmtree(staging, ignore_errors=True)
    else:
        staging.replace(cache_dir)

    print(cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
