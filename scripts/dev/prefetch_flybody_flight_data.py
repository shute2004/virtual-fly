#!/usr/bin/env python3
"""Fetch the official FlyBody measured baseline wing-beat pattern.

The FlyBody source historically mapped ``flight-imitation-dataset`` to Figshare
file 51196859 and assumed that URL returned a ZIP archive.  That assumption is
no longer reliable in practice.  This prefetcher therefore discovers the files
attached to the published Figshare article/version through the official API,
then downloads only the smallest plausible flight-related object and extracts
``wing_pattern_fmech.npy`` from it.

Nothing downloaded here is committed to the repository; the measured upstream
binary is cached under ``artifacts/flybody-data``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.request
import zipfile

import numpy as np


ARTICLE_ID = 25309105
ARTICLE_VERSION = 4
ARTICLE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}"
LEGACY_URL = "https://janelia.figshare.com/ndownloader/files/51196859"
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


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "virtual-fly/1.0 FlyBody data prefetch",
            "Accept": "application/json, application/octet-stream, */*",
        },
    )


def fetch_json(url: str, timeout_s: float) -> dict[str, object]:
    with urllib.request.urlopen(request(url), timeout=timeout_s) as response:
        payload = response.read()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Figshare API did not return JSON: {url}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected Figshare API payload type: {type(value).__name__}")
    return value


def download(url: str, path: Path, timeout_s: float) -> dict[str, str]:
    print(f"downloading={url}", flush=True)
    with urllib.request.urlopen(request(url), timeout=timeout_s) as response, path.open("wb") as dst:
        headers = {
            "content_type": response.headers.get("Content-Type", ""),
            "content_disposition": response.headers.get("Content-Disposition", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "final_url": response.geturl(),
        }
        shutil.copyfileobj(response, dst, length=1024 * 1024)
    return headers


def extract_target(source: Path, output: Path, tmp_dir: Path) -> tuple[int, int] | None:
    # Figshare may expose the .npy itself as a file rather than an archive.
    with source.open("rb") as handle:
        magic = handle.read(8)
    if magic.startswith(b"\x93NUMPY"):
        rows, cols = validate_pattern(source)
        shutil.copy2(source, output)
        return rows, cols

    if not zipfile.is_zipfile(source):
        return None

    with zipfile.ZipFile(source) as zf:
        matches = [name for name in zf.namelist() if Path(name).name == TARGET_BASENAME]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {TARGET_BASENAME} in {source.name}, found {matches}"
            )
        staged = tmp_dir / TARGET_BASENAME
        with zf.open(matches[0]) as src, staged.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        rows, cols = validate_pattern(staged)
        staged.replace(output)
        return rows, cols


def file_download_url(entry: dict[str, object]) -> str | None:
    for key in ("download_url", "url_private_api", "url"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    file_id = entry.get("id")
    if isinstance(file_id, int) or (isinstance(file_id, str) and file_id.isdigit()):
        return f"https://janelia.figshare.com/ndownloader/files/{file_id}"
    return None


def candidate_rank(entry: dict[str, object]) -> tuple[int, int, str]:
    name = str(entry.get("name", "")).lower()
    size = int(entry.get("size", 2**63 - 1) or 2**63 - 1)
    if Path(name).name == TARGET_BASENAME:
        priority = 0
    elif "flight" in name and name.endswith(".zip"):
        priority = 1
    elif "flight" in name:
        priority = 2
    elif name.endswith(".zip"):
        priority = 3
    else:
        priority = 4
    return priority, size, name


def discover_figshare_files(timeout_s: float) -> list[dict[str, object]]:
    metadata = fetch_json(ARTICLE_API, timeout_s)
    raw_files = metadata.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("Figshare article metadata contains no files list")
    files = [row for row in raw_files if isinstance(row, dict)]
    if not files:
        raise RuntimeError("Figshare article metadata returned an empty files list")
    return sorted(files, key=candidate_rank)


def main() -> int:
    output = Path(os.environ.get("VF_FLYBODY_WING_PATTERN", str(DEFAULT_OUTPUT)))
    timeout_s = float(os.environ.get("VF_FLYBODY_FLIGHT_DATA_TIMEOUT_S", "180"))
    override_url = os.environ.get("VF_FLYBODY_FLIGHT_DATA_URL")
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

    with tempfile.TemporaryDirectory(prefix="virtual-fly-flight-data-") as tmp_name:
        tmp_dir = Path(tmp_name)

        # Explicit override is useful for mirrors/offline reproductions.
        if override_url:
            source = tmp_dir / "override-download"
            headers = download(override_url, source, timeout_s)
            extracted = extract_target(source, output, tmp_dir)
            if extracted is None:
                raise RuntimeError(
                    "VF_FLYBODY_FLIGHT_DATA_URL did not provide a .npy or ZIP "
                    f"containing {TARGET_BASENAME}; headers={headers}"
                )
            rows, cols = extracted
        else:
            files = discover_figshare_files(timeout_s)
            print(f"figshare_article={ARTICLE_ID} version={ARTICLE_VERSION}")
            print("figshare_files=" + ", ".join(
                f"{row.get('name','?')}({row.get('size','?')})" for row in files
            ))

            attempted: list[str] = []
            rows = cols = 0
            found = False
            # Avoid accidentally downloading multi-GB unrelated files.  Flight data
            # and the wing pattern should rank ahead of generic archives.
            for index, entry in enumerate(files):
                priority, size, name = candidate_rank(entry)
                if priority >= 4:
                    continue
                if size > 512 * 1024 * 1024:
                    attempted.append(f"{name}: skipped-large({size})")
                    continue
                url = file_download_url(entry)
                if url is None:
                    attempted.append(f"{name}: no-download-url")
                    continue
                source = tmp_dir / f"candidate-{index}"
                headers = download(url, source, timeout_s)
                attempted.append(f"{name}: {headers.get('content_type','')}")
                extracted = extract_target(source, output, tmp_dir)
                if extracted is not None:
                    rows, cols = extracted
                    found = True
                    print(f"selected_figshare_file={name}")
                    break

            if not found:
                raise RuntimeError(
                    f"could not locate {TARGET_BASENAME} in Figshare article version "
                    f"{ARTICLE_VERSION}; attempted={attempted}. The historical FlyBody "
                    f"URL was {LEGACY_URL}. Do not download the 4.05 GB article bundle "
                    "blindly; inspect the printed Figshare file list first."
                )

    print(f"wing_pattern={output}")
    print(f"wing_pattern_shape=({rows},{cols})")
    print("flybody_flight_data=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
