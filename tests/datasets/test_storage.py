from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from src.datasets.storage import (
    CorruptRecordError,
    LmdbShardWriter,
    RecordCodec,
    RecoveredShardError,
    ShardedRecordStore,
    StoreMetadata,
    StoreSchemaError,
    UnsupportedArrayError,
    write_store_metadata,
)


def test_record_codec_round_trip_preserves_arrays_and_nested_metadata() -> None:
    codec = RecordCodec(compression_level=1)
    record = {
        "sample_id": "pcqm:17",
        "source_index": 17,
        "input_ids": np.array([2, 9, 3], dtype=np.int32),
        "graph": {
            "node_feat": np.array([[5, 0], [7, 1]], dtype=np.int16),
            "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int32),
        },
        "quality": {"sdf_aligned": True, "reason": None},
    }

    decoded = codec.decode(codec.encode(record))

    assert decoded["sample_id"] == "pcqm:17"
    assert decoded["quality"] == {"sdf_aligned": True, "reason": None}
    np.testing.assert_array_equal(decoded["input_ids"], np.array([2, 9, 3], dtype=np.int32))
    np.testing.assert_array_equal(
        decoded["graph"]["edge_index"],
        np.array([[0, 1], [1, 0]], dtype=np.int32),
    )


def test_record_codec_rejects_object_arrays() -> None:
    codec = RecordCodec()

    with pytest.raises(UnsupportedArrayError, match="object"):
        codec.encode({"unsafe": np.array([{"value": 1}], dtype=object)})


def test_record_codec_detects_payload_corruption() -> None:
    codec = RecordCodec(compression_level=1)
    payload = bytearray(codec.encode({"value": np.array([1, 2, 3], dtype=np.int8)}))
    payload[-1] ^= 0x01

    with pytest.raises(CorruptRecordError):
        codec.decode(bytes(payload))


def test_sharded_store_reads_records_by_global_record_index(tmp_path) -> None:
    store_dir = tmp_path / "store"
    codec = RecordCodec(compression_level=1)

    with LmdbShardWriter(
        store_dir=store_dir,
        shard_id=0,
        start_index=0,
        expected_records=2,
        map_size=8 * 1024 * 1024,
        codec=codec,
    ) as writer:
        writer.put(0, {"sample_id": "pcqm:0", "value": np.array([10], dtype=np.int16)})
        writer.put(1, {"sample_id": "pcqm:1", "value": np.array([11], dtype=np.int16)})

    metadata = StoreMetadata(
        schema_version=1,
        record_count=2,
        records_per_shard=2,
        modalities=("1d",),
        tokenizer_sha256="a" * 64,
        tokenizer_vocab_size=32,
        shards=("shard-000000.lmdb",),
    )
    write_store_metadata(store_dir, metadata)

    store = ShardedRecordStore(store_dir)
    try:
        assert store[1]["sample_id"] == "pcqm:1"
        np.testing.assert_array_equal(store[1]["value"], np.array([11], dtype=np.int16))
    finally:
        store.close()

    sidecar = json.loads((store_dir / "shard-000000.json").read_text(encoding="utf-8"))
    assert sidecar["record_count"] == 2
    assert len(sidecar["sha256"]) == 64


def test_shard_writer_refuses_non_contiguous_or_duplicate_indices(tmp_path) -> None:
    writer = LmdbShardWriter(
        store_dir=tmp_path / "store",
        shard_id=0,
        start_index=4,
        expected_records=2,
        map_size=8 * 1024 * 1024,
    )
    try:
        writer.put(4, {"sample_id": "pcqm:4"})
        with pytest.raises(ValueError, match="expected record index 5"):
            writer.put(4, {"sample_id": "duplicate"})
    finally:
        writer.abort()


def test_store_metadata_rejects_duplicate_modalities_and_unknown_record_schema() -> None:
    with pytest.raises(StoreSchemaError, match="重复"):
        StoreMetadata(
            schema_version=1,
            record_count=0,
            records_per_shard=2,
            modalities=("1d", "1d"),
            tokenizer_sha256="a" * 64,
            tokenizer_vocab_size=32,
            shards=(),
        ).validate()

    with pytest.raises(StoreSchemaError, match="record_schema"):
        StoreMetadata(
            schema_version=1,
            record_count=0,
            records_per_shard=2,
            modalities=(),
            tokenizer_sha256="",
            tokenizer_vocab_size=0,
            shards=(),
            record_schema="semmol.multimodal.v2",
        ).validate()


def test_store_metadata_requires_vocab_size_for_1d_records() -> None:
    with pytest.raises(StoreSchemaError, match="tokenizer_vocab_size"):
        StoreMetadata(
            schema_version=1,
            record_count=0,
            records_per_shard=2,
            modalities=("1d",),
            tokenizer_sha256="a" * 64,
            tokenizer_vocab_size=0,
            shards=(),
        ).validate()


def test_shard_writer_recovers_sidecar_after_interrupted_publication(tmp_path) -> None:
    store_dir = tmp_path / "store"
    with LmdbShardWriter(
        store_dir=store_dir,
        shard_id=0,
        start_index=0,
        expected_records=1,
        map_size=8 * 1024 * 1024,
    ) as writer:
        writer.put(0, {"sample_id": "pcqm:0"})
    sidecar = store_dir / "shard-000000.json"
    sidecar.unlink()

    with pytest.raises(RecoveredShardError, match="恢复"):
        LmdbShardWriter(
            store_dir=store_dir,
            shard_id=0,
            start_index=0,
            expected_records=1,
            map_size=8 * 1024 * 1024,
        )

    recovered = json.loads(sidecar.read_text(encoding="utf-8"))
    assert recovered["record_count"] == 1
    assert len(recovered["sha256"]) == 64


def test_sharded_store_bounds_open_readers_and_resets_after_pid_change(
    tmp_path,
    monkeypatch,
) -> None:
    store_dir = tmp_path / "store"
    for shard_id in range(3):
        with LmdbShardWriter(
            store_dir=store_dir,
            shard_id=shard_id,
            start_index=shard_id,
            expected_records=1,
            map_size=8 * 1024 * 1024,
        ) as writer:
            writer.put(shard_id, {"sample_id": f"pcqm:{shard_id}"})
    write_store_metadata(
        store_dir,
        StoreMetadata(
            schema_version=1,
            record_count=3,
            records_per_shard=1,
            modalities=(),
            tokenizer_sha256="",
            tokenizer_vocab_size=0,
            shards=tuple(
                f"shard-{shard_id:06d}.lmdb"
                for shard_id in range(3)
            ),
        ),
    )

    store = ShardedRecordStore(store_dir, max_open_shards=1)
    assert store[0]["sample_id"] == "pcqm:0"
    first_reader = next(iter(store._readers.values()))
    assert store[1]["sample_id"] == "pcqm:1"
    assert len(store._readers) == 1
    assert first_reader._environment is None

    original_pid = store._owner_pid
    monkeypatch.setattr("src.datasets.storage.os.getpid", lambda: original_pid + 1)
    inherited_reader = next(iter(store._readers.values()))
    assert store[2]["sample_id"] == "pcqm:2"
    assert inherited_reader._environment is None
    assert store._owner_pid == original_pid + 1


def test_sharded_store_can_be_pickled_for_spawn_workers(tmp_path) -> None:
    store_dir = tmp_path / "store"
    with LmdbShardWriter(
        store_dir=store_dir,
        shard_id=0,
        start_index=0,
        expected_records=1,
        map_size=8 * 1024 * 1024,
    ) as writer:
        writer.put(0, {"sample_id": "pcqm:0"})
    write_store_metadata(
        store_dir,
        StoreMetadata(
            schema_version=1,
            record_count=1,
            records_per_shard=1,
            modalities=(),
            tokenizer_sha256="",
            tokenizer_vocab_size=0,
            shards=("shard-000000.lmdb",),
        ),
    )
    store = ShardedRecordStore(store_dir)
    assert store[0]["sample_id"] == "pcqm:0"

    restored = pickle.loads(pickle.dumps(store))

    assert restored[0]["sample_id"] == "pcqm:0"
