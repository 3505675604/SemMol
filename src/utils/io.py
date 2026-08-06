"""Atomic writes, checksums, and JSONL utilities for preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Union


PathLike = Union[str, os.PathLike[str]]


def _fsync_directory(directory: Path) -> None:
    """Durably persist a directory entry on the Linux training server."""

    if os.name == "nt":
        # Windows does not expose a portable directory fsync through os.open.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_output_path(
    target: PathLike,
    *,
    overwrite: bool = False,
) -> Iterator[Path]:
    """Create a same-directory temporary path for a single-file artifact and publish it atomically on success.

    The caller is responsible for writing complete content to the yielded path. On
    failure, only the temporary file created by this invocation is removed.
    Existing targets are not overwritten by default; they are replaced only when
    ``overwrite=True`` is explicitly supplied.
    """

    target_path = Path(target).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Target already exists; refusing to overwrite: {target_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.tmp-",
        dir=str(target_path.parent),
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name).resolve()
    try:
        yield temporary_path
        if not temporary_path.is_file():
            raise FileNotFoundError(f"Caller did not create the temporary artifact: {temporary_path}")
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, target_path)
        else:
            # A second existence check followed by os.replace is racy: a
            # concurrent writer can publish between those operations and be
            # overwritten.  Creating a hard link in the same directory is an
            # atomic no-replace publication and raises FileExistsError when a
            # competing target already exists.
            os.link(temporary_path, target_path)
            temporary_path.unlink()
        _fsync_directory(target_path.parent)
    except BaseException:
        if temporary_path.exists() and temporary_path.parent == target_path.parent:
            temporary_path.unlink()
        raise


def atomic_write_json(
    target: PathLike,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    target_path = Path(target).resolve()
    with atomic_output_path(target_path, overwrite=overwrite) as temporary_path:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return target_path


def atomic_write_jsonl(
    target: PathLike,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    target_path = Path(target).resolve()
    with atomic_output_path(target_path, overwrite=overwrite) as temporary_path:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return target_path


def sha256_file(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
