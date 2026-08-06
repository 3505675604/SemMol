from __future__ import annotations

import json

import pytest

from src.utils.io import atomic_output_path, atomic_write_json, sha256_file


def test_atomic_output_path_publishes_complete_file(tmp_path) -> None:
    target = tmp_path / "artifact.txt"

    with atomic_output_path(target) as temporary:
        temporary.write_text("complete", encoding="utf-8")
        assert not target.exists()

    assert target.read_text(encoding="utf-8") == "complete"


def test_atomic_output_path_removes_temporary_file_after_failure(tmp_path) -> None:
    target = tmp_path / "artifact.txt"

    with pytest.raises(RuntimeError, match="stop"):
        with atomic_output_path(target) as temporary:
            temporary.write_text("partial", encoding="utf-8")
            raise RuntimeError("stop")

    assert not target.exists()
    assert list(tmp_path.glob(".artifact.txt.tmp-*")) == []


def test_atomic_json_and_sha256_are_deterministic(tmp_path) -> None:
    target = tmp_path / "metadata.json"

    atomic_write_json(target, {"z": 1, "a": ["中文", True]})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "a": ["中文", True],
        "z": 1,
    }
    assert sha256_file(target) == sha256_file(target)


def test_no_overwrite_publication_cannot_replace_a_concurrent_writer(
    tmp_path,
) -> None:
    target = tmp_path / "artifact.txt"

    with pytest.raises(FileExistsError):
        with atomic_output_path(target, overwrite=False) as temporary:
            temporary.write_text("ours", encoding="utf-8")
            target.write_text("concurrent", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "concurrent"
    assert list(tmp_path.glob(".artifact.txt.tmp-*")) == []
