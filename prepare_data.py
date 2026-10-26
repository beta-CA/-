"""Prepare verified CIFAR-10 data with resumable parallel HTTP range downloads.

Download transfer code reused from the earlier Cutout experiment scaffold.
Dataset metadata, MD5 checking, and extraction come from torchvision.
Existing verified CIFAR-10 files may be copied; no model or GPU is initialized.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.request import Request, urlopen
import uuid


# This is the original University of Toronto host, not a third-party mirror.
DOWNLOAD_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
ARCHIVE_BYTES = 170_498_071
CHUNK_BYTES = 4 * 1024 * 1024
DOWNLOAD_WORKERS = 8
SOCKET_TIMEOUT = 30
MAX_ATTEMPTS = 5


def say(message: str) -> None:
    print(message, flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def chunk_path(parts: Path, index: int) -> Path:
    return parts / f"chunk-{index:04d}.bin"


def download_archive(
    data_root: Path,
    filename: str,
    md5: str,
    check_integrity,
    *,
    url: str = DOWNLOAD_URL,
    total: int = ARCHIVE_BYTES,
    chunk_bytes: int = CHUNK_BYTES,
    workers: int = DOWNLOAD_WORKERS,
) -> Path:
    """Keep the old archive until a complete replacement passes official MD5."""
    archive = data_root / filename
    parts = data_root / f".{filename}.parts"
    identity = {"url": url, "bytes": total, "md5": md5, "chunk_bytes": chunk_bytes}
    ranges = [(i, start, min(start + chunk_bytes, total) - 1)
              for i, start in enumerate(range(0, total, chunk_bytes))]

    # One fresh rebuild handles an invalid cached chunk or untrusted old prefix.
    for build_attempt in range(2):
        if parts.exists():
            manifest = parts / "manifest.json"
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise RuntimeError(f"Download cache metadata differs: {parts}. Preserve it and use a new cache directory.")
        else:
            parts.mkdir(parents=True)
            atomic_json(parts / "manifest.json", identity)

        # Reuse complete blocks only. Their final authority is whole-file MD5.
        if build_attempt == 0 and archive.is_file() and 0 < archive.stat().st_size < total:
            prefix_size = archive.stat().st_size
            with archive.open("rb") as prefix:
                for index, start, end in ranges:
                    size = end - start + 1
                    if end >= prefix_size:
                        break
                    destination = chunk_path(parts, index)
                    if destination.is_file() and destination.stat().st_size == size:
                        continue
                    prefix.seek(start)
                    block = prefix.read(size)
                    if len(block) != size:
                        raise RuntimeError("The original partial archive changed while reading it")
                    temporary = destination.with_suffix(".tmp")
                    temporary.write_bytes(block)
                    temporary.replace(destination)

        completed = {}
        missing = []
        for index, start, end in ranges:
            destination = chunk_path(parts, index)
            size = end - start + 1
            if destination.is_file() and destination.stat().st_size == size:
                completed[index] = size
            else:
                missing.append((index, start, end))
        progress = dict(completed)
        lock = threading.Lock()
        cancelled = threading.Event()

        def download_one(index: int, start: int, end: int) -> None:
            expected_size = end - start + 1
            destination = chunk_path(parts, index)
            temporary = destination.with_suffix(".tmp")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if cancelled.is_set():
                    raise RuntimeError("Download cancelled after another chunk failed")
                with lock:
                    progress[index] = 0
                try:
                    request = Request(url, headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                        "User-Agent": "Cutout-CIFAR10-reproduction/1.0",
                    })
                    with urlopen(request, timeout=SOCKET_TIMEOUT) as response:
                        expected_range = f"bytes {start}-{end}/{total}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                            raise RuntimeError(f"Server did not honor the exact range {expected_range}")
                        received = 0
                        with temporary.open("wb") as stream:
                            while received < expected_size:
                                if cancelled.is_set():
                                    raise RuntimeError("Download cancelled after another chunk failed")
                                block = response.read(min(65536, expected_size - received))
                                if not block:
                                    raise RuntimeError(f"Incomplete range: {received}/{expected_size} bytes")
                                stream.write(block)
                                received += len(block)
                                with lock:
                                    progress[index] = received
                            if response.read(1):
                                raise RuntimeError("Server sent bytes outside the requested range")
                            stream.flush()
                            os.fsync(stream.fileno())
                    temporary.replace(destination)
                    return
                except Exception as error:
                    if cancelled.is_set() or attempt == MAX_ATTEMPTS:
                        raise RuntimeError(f"Chunk {index} failed after {attempt} attempt(s): {error}") from error
                    with lock:
                        progress[index] = 0
                    say(f"Chunk {index}: retry {attempt}/{MAX_ATTEMPTS - 1} after {type(error).__name__}: {error}")
                    if cancelled.wait(min(2 ** (attempt - 1), 8)):
                        raise RuntimeError("Download cancelled") from error

        say(f"CIFAR-10 download: {sum(completed.values()):,}/{total:,} bytes cached; "
            f"{len(missing)} chunks remain; up to {workers} parallel transfers.")
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cifar-download")
        futures = {executor.submit(download_one, *item) for item in missing}
        try:
            pending = futures
            while pending:
                finished, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                with lock:
                    current = sum(progress.values())
                say(f"Download {100 * current / total:5.1f}% "
                    f"({current / 1024**2:.1f}/{total / 1024**2:.1f} MiB), "
                    f"elapsed {time.monotonic() - started:.0f}s")
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        assembled = parts / "assembled.tar.gz"
        say("Assembling chunks and checking the official archive MD5...")
        with assembled.open("wb") as stream:
            for index, start, end in ranges:
                source = chunk_path(parts, index)
                if not source.is_file() or source.stat().st_size != end - start + 1:
                    raise RuntimeError(f"Missing or incomplete chunk: {source}")
                with source.open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        if assembled.stat().st_size == total and check_integrity(str(assembled), md5):
            assembled.replace(archive)
            say(f"Official archive MD5 verified: {archive}")
            return archive

        quarantine = parts.with_name(parts.name + ".bad-md5-" +
                                     datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
        parts.rename(quarantine)
        if build_attempt == 0:
            say(f"Archive MD5 mismatch. Preserved failed cache at {quarantine}; "
                "retrying once with every chunk fetched fresh from the official host.")
        else:
            raise RuntimeError(f"Official archive MD5 still failed after a fresh download; "
                               f"original archive preserved, failed cache at {quarantine}")
    raise RuntimeError("Download did not produce a verified archive")



def prepare_data(project_root: Path, reuse_root: Path | None = None) -> Path:
    """Verify project-local data, optionally copy verified files, or resume download."""
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity, extract_archive

    data_root = project_root.resolve() / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    checks = list(CIFAR10.train_list) + list(CIFAR10.test_list) + [
        (CIFAR10.meta["filename"], CIFAR10.meta["md5"])]

    def data_valid(directory: Path) -> bool:
        return all(check_integrity(str(directory / CIFAR10.base_folder / name), md5)
                   for name, md5 in checks)

    if data_valid(data_root):
        say("Project-local CIFAR-10 files already pass all 7 official MD5 checks.")
        return data_root
    if reuse_root is not None:
        reuse_root = Path(reuse_root).resolve()
        if reuse_root != data_root and data_valid(reuse_root):
            destination_folder = data_root / CIFAR10.base_folder
            destination_folder.mkdir(parents=True, exist_ok=True)
            say(f"Reusing verified CIFAR-10 files from {reuse_root}; source files remain unchanged.")
            for name, md5 in checks:
                target = destination_folder / name
                if check_integrity(str(target), md5):
                    continue
                temporary = target.with_name(target.name + ".copying")
                shutil.copyfile(reuse_root / CIFAR10.base_folder / name, temporary)
                if not check_integrity(str(temporary), md5):
                    raise RuntimeError(f"Copied data failed the official checksum: {name}")
                temporary.replace(target)
            if not data_valid(data_root):
                raise RuntimeError("Project-local copied dataset failed final integrity checks")
            say("All project-local copies verified. No download was necessary.")
            return data_root
        say("Reusable extracted data were unavailable or failed checks; checking the official archive.")
    archive = data_root / CIFAR10.filename
    if not check_integrity(str(archive), CIFAR10.tgz_md5):
        if reuse_root is not None and check_integrity(str(reuse_root / CIFAR10.filename), CIFAR10.tgz_md5):
            temporary = archive.with_suffix(archive.suffix + ".copying")
            shutil.copyfile(reuse_root / CIFAR10.filename, temporary)
            if not check_integrity(str(temporary), CIFAR10.tgz_md5):
                raise RuntimeError("Copied archive failed its official checksum")
            temporary.replace(archive)
        else:
            archive = download_archive(data_root, CIFAR10.filename, CIFAR10.tgz_md5, check_integrity)
    say("Extracting the verified official archive with torchvision...")
    extract_archive(str(archive), str(data_root), remove_finished=False)
    if not data_valid(data_root):
        raise RuntimeError("Extracted files failed official MD5 checks; training was not started")
    say("CIFAR-10 preparation complete; all 7 dataset/metadata files verified.")
    return data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reuse-data", type=Path, default=None,
                        help="Optional existing CIFAR-10 data root; only verified files are copied")
    args = parser.parse_args()
    prepare_data(args.project_root, args.reuse_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Preparation interrupted. Complete download chunks are retained.", file=sys.stderr, flush=True)
        sys.exit(130)
    except Exception as error:
        print(f"Data preparation failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
