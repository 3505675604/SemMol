"""Secure, recoverable sharded LMDB record storage.

Training samples are not serialized via pickle. Each record is first converted into a MsgPack-compatible data structure.
NumPy arrays are encoded with ``dtype + shape + raw bytes`` and compressed by Zstandard. The SHA-256 digest of the uncompressed payload is stored in the record header.
LMDB shards are only created by a single writing process. After completion, they are
published through an atomic rename; the training process is always read-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import shutil
import struct
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import lmdb
except ImportError:  # pragma: no cover - explicit error given by server environment check
    lmdb = None

try:
    import msgpack
except ImportError:  # pragma: no cover
    msgpack = None

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None


PathLike = Union[str, os.PathLike[str]]
STORE_METADATA_NAME = "store.json"
STORE_SCHEMA_VERSION = 1
RECORD_SCHEMA = "semmol.multimodal.v1"
RECORD_MAGIC = b"SEMMOLRC1"
RECORD_DIGEST_SIZE = hashlib.sha256().digest_size
ARRAY_MARKER = "__semmol_ndarray_v1__"
SHARD_METADATA_KEY = b"__semmol_shard_metadata__"
RECORD_KEY_STRUCT = struct.Struct(">Q")


class StorageError(RuntimeError):
    """SemMol records storage base exceptions."""


class MissingStorageDependencyError(StorageError):
    """The server environment is missing record storage dependencies."""


class UnsupportedArrayError(StorageError):
    """The record contains an array that cannot be safely and stably serialized."""


class CorruptRecordError(StorageError):
    """Record payload, checksum, or array metadata corruption."""


class StoreSchemaError(StorageError):
    """The storage metadata does not satisfy the current schema."""


class IncompleteShardError(StorageError):
    """The number of shard writes is inconsistent with the declared expected number."""


class RecoveredShardError(StorageError):
    """Detected and restored shards that were published at the time of the last outage but were missing sidecars."""


def _require_codec_dependencies() -> None:
    missing = []
    if msgpack is None:
        missing.append("msgpack")
    if zstandard is None:
        missing.append("zstandard")
    if missing:
        joined = ", ".join(missing)
        raise MissingStorageDependencyError(
            f"Missing record codec dependencies: {joined}; install them from the server requirements.txt"
        )


def _require_lmdb() -> None:
    if lmdb is None:
        raise MissingStorageDependencyError(
            "Lmdb is missing; please install lmdb according to server requirements.txt"
        )


def _record_key(record_index: int) -> bytes:
    if record_index < 0:
        raise ValueError(f"record_index must be non-negative, got {record_index}")
    if record_index > (2**64 - 1):
        raise ValueError(f"record_index exceeds uint64: {record_index}")
    return RECORD_KEY_STRUCT.pack(record_index)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory entry changes on server platforms that support directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".json",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(f"Target already exists; refusing to overwrite: {path}") from None
        temporary_path.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


class RecordCodec:
    """MsgPack + Zstandard recording codec.

    Args:
        compression_level: Zstandard compression level.
        max_record_bytes: The maximum number of bytes allowed in a single uncompressed record, which limits memory allocation for malformed or oversized payloads.
        max_array_elements: The maximum number of elements allowed in a single NumPy array.
    """

    def __init__(
        self,
        compression_level: int = 3,
        max_record_bytes: int = 512 * 1024 * 1024,
        max_array_elements: int = 256 * 1024 * 1024,
    ) -> None:
        _require_codec_dependencies()
        if not -7 <= compression_level <= 22:
            raise ValueError("Zstandard compression_level must be in [-7, 22]")
        if max_record_bytes <= 0 or max_array_elements <= 0:
            raise ValueError("Record and array caps must be positive integers")
        self.compression_level = int(compression_level)
        self.max_record_bytes = int(max_record_bytes)
        self.max_array_elements = int(max_array_elements)
        self._compressor = zstandard.ZstdCompressor(level=self.compression_level)
        self._decompressor = zstandard.ZstdDecompressor()

    def _to_msgpack(self, value: Any, path: str = "$") -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.kind == "O":
                raise UnsupportedArrayError(f"{path} uses object dtype, which cannot be serialized")
            if value.dtype.kind not in {"b", "i", "u", "f"}:
                raise UnsupportedArrayError(
                    f"{path} has dtype {value.dtype}, which is outside the allowed bool/int/uint/float range"
                )
            if value.size > self.max_array_elements:
                raise UnsupportedArrayError(
                    f"{path} has {value.size} elements, exceeding the limit of {self.max_array_elements}"
                )
            contiguous = np.ascontiguousarray(value)
            return {
                ARRAY_MARKER: True,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data": contiguous.tobytes(order="C"),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            converted: Dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} record dictionary keys must be str, got {type(key)!r}")
                if key == ARRAY_MARKER:
                    raise ValueError(f"{path} uses the reserved key {ARRAY_MARKER!r}")
                converted[key] = self._to_msgpack(item, f"{path}.{key}")
            return converted
        if isinstance(value, (list, tuple)):
            return [
                self._to_msgpack(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if value is None or isinstance(value, (str, bytes, bool, int, float)):
            return value
        raise TypeError(f"{path} contains unsupported type {type(value)!r}")

    def _from_msgpack(self, value: Any, path: str = "$") -> Any:
        if isinstance(value, dict):
            if value.get(ARRAY_MARKER) is True:
                required = {ARRAY_MARKER, "dtype", "shape", "data"}
                if set(value) != required:
                    raise CorruptRecordError(f"{path} has an invalid set of array fields")
                try:
                    dtype = np.dtype(value["dtype"])
                    shape = tuple(int(dim) for dim in value["shape"])
                except (TypeError, ValueError) as exc:
                    raise CorruptRecordError(f"{path} has invalid dtype/shape") from exc
                if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
                    raise CorruptRecordError(f"{path} contains disallowed dtype {dtype}")
                if any(dim < 0 for dim in shape):
                    raise CorruptRecordError(f"{path} contains negative dimensions {shape}")
                element_count = math.prod(shape)
                if element_count > self.max_array_elements:
                    raise CorruptRecordError(
                        f"{path} has {element_count} elements, exceeding the limit of {self.max_array_elements}"
                    )
                raw = value["data"]
                if not isinstance(raw, bytes):
                    raise CorruptRecordError(f"{path} array data is not bytes")
                expected_bytes = element_count * dtype.itemsize
                if len(raw) != expected_bytes:
                    raise CorruptRecordError(
                        f"{path} array byte count {len(raw)} != expected {expected_bytes}"
                    )
                return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            converted: Dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CorruptRecordError(
                        f"{path} record dictionary keys must be str"
                    )
                if key in converted:
                    raise CorruptRecordError(
                        f"{path} record dictionary contains duplicate key {key!r}"
                    )
                converted[key] = self._from_msgpack(item, f"{path}.{key}")
            return converted
        if isinstance(value, list):
            return [
                self._from_msgpack(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        return value

    def encode(self, record: Mapping[str, Any]) -> bytes:
        if not isinstance(record, Mapping):
            raise TypeError("The record must be Mapping[str, Any]")
        transformed = self._to_msgpack(record)
        packed = msgpack.packb(transformed, use_bin_type=True, strict_types=True)
        if len(packed) > self.max_record_bytes:
            raise StorageError(
                f"Uncompressed record of {len(packed)} bytes exceeds the limit of {self.max_record_bytes}"
            )
        digest = hashlib.sha256(packed).digest()
        compressed = self._compressor.compress(packed)
        return RECORD_MAGIC + digest + compressed

    def decode(self, payload: bytes) -> Dict[str, Any]:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        header_size = len(RECORD_MAGIC) + RECORD_DIGEST_SIZE
        if len(payload) <= header_size or not payload.startswith(RECORD_MAGIC):
            raise CorruptRecordError("Invalid record header or magic")
        expected_digest = payload[len(RECORD_MAGIC):header_size]
        compressed = payload[header_size:]
        try:
            packed = self._decompressor.decompress(
                compressed,
                max_output_size=self.max_record_bytes,
            )
        except Exception as exc:
            raise CorruptRecordError("Zstandard records decompression failure") from exc
        if len(packed) > self.max_record_bytes:
            raise CorruptRecordError("Decompressed record exceeds max_record_bytes")
        actual_digest = hashlib.sha256(packed).digest()
        if not hmac.compare_digest(expected_digest, actual_digest):
            raise CorruptRecordError("Record SHA-256 digest verification failed")
        try:
            unpacked = msgpack.unpackb(
                packed,
                raw=False,
                strict_map_key=True,
                max_str_len=self.max_record_bytes,
                max_bin_len=self.max_record_bytes,
                max_array_len=self.max_record_bytes,
                max_map_len=self.max_record_bytes,
                max_ext_len=self.max_record_bytes,
            )
        except Exception as exc:
            raise CorruptRecordError("MsgPack record parsing failed") from exc
        decoded = self._from_msgpack(unpacked)
        if not isinstance(decoded, dict):
            raise CorruptRecordError("The record root node must be a dict")
        return decoded

    def __getstate__(self) -> Dict[str, int]:
        """Only the configuration is serialized so that the Windows spawn worker does not touch the C extension handle."""

        return {
            "compression_level": self.compression_level,
            "max_record_bytes": self.max_record_bytes,
            "max_array_elements": self.max_array_elements,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(
            compression_level=int(state["compression_level"]),
            max_record_bytes=int(state["max_record_bytes"]),
            max_array_elements=int(state["max_array_elements"]),
        )


@dataclass(frozen=True)
class StoreMetadata:
    """Immutable index information for the entire shard record library."""

    schema_version: int
    record_count: int
    records_per_shard: int
    modalities: Tuple[str, ...]
    tokenizer_sha256: str
    shards: Tuple[str, ...]
    tokenizer_vocab_size: int = 0
    record_schema: str = RECORD_SCHEMA

    def validate(self) -> None:
        integer_fields = {
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "records_per_shard": self.records_per_shard,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise StoreSchemaError(f"{name} must be an integer")
        if self.schema_version != STORE_SCHEMA_VERSION:
            raise StoreSchemaError(
                f"store schema_version={self.schema_version}, "
                f"the code only supports {STORE_SCHEMA_VERSION}"
            )
        if self.record_count < 0:
            raise StoreSchemaError("record_count cannot be negative")
        if self.records_per_shard <= 0:
            raise StoreSchemaError("records_per_shard must be a positive integer")
        if not isinstance(self.modalities, tuple) or any(
            not isinstance(item, str) for item in self.modalities
        ):
            raise StoreSchemaError("modalities must be a string tuple")
        if not isinstance(self.shards, tuple) or any(
            not isinstance(item, str) or not item for item in self.shards
        ):
            raise StoreSchemaError("shards must be a non-empty string tuple")
        if not isinstance(self.tokenizer_sha256, str):
            raise StoreSchemaError("tokenizer_sha256 must be a string")
        if not isinstance(self.record_schema, str):
            raise StoreSchemaError("record_schema must be a string")
        expected_shards = (
            math.ceil(self.record_count / self.records_per_shard)
            if self.record_count
            else 0
        )
        if len(self.shards) != expected_shards:
            raise StoreSchemaError(
                f"shard count {len(self.shards)} != expected {expected_shards}"
            )
        if len(set(self.shards)) != len(self.shards):
            raise StoreSchemaError("Duplicate file names in shards")
        if self.record_schema != RECORD_SCHEMA:
            raise StoreSchemaError(
                f"record_schema={self.record_schema!r}, the code only supports {RECORD_SCHEMA!r}"
            )
        allowed_modalities = {"1d", "2d", "3d", "qm"}
        unknown = set(self.modalities) - allowed_modalities
        if unknown:
            raise StoreSchemaError(f"Unknown modalities: {sorted(unknown)}")
        if len(set(self.modalities)) != len(self.modalities):
            raise StoreSchemaError("Duplicate modalities exist in modalities")
        if "1d" in self.modalities and not self.tokenizer_sha256:
            raise StoreSchemaError("Stores containing 1d modal must log tokenizer_sha256")
        if self.tokenizer_sha256 and (
            len(self.tokenizer_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.tokenizer_sha256
            )
        ):
            raise StoreSchemaError("tokenizer_sha256 must be a 64-character lowercase hexadecimal digest")
        if self.tokenizer_vocab_size < 0:
            raise StoreSchemaError("tokenizer_vocab_size cannot be negative")
        if "1d" in self.modalities and self.tokenizer_vocab_size <= 0:
            raise StoreSchemaError(
                "Stores containing 1d modal must record positive tokenizer_vocab_size"
            )
        if "1d" not in self.modalities and (
            bool(self.tokenizer_sha256) != (self.tokenizer_vocab_size > 0)
        ):
            raise StoreSchemaError(
                "tokenizer_sha256 and tokenizer_vocab_size must both exist or be empty at the same time"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "records_per_shard": self.records_per_shard,
            "modalities": list(self.modalities),
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "shards": list(self.shards),
            "record_schema": self.record_schema,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoreMetadata":
        required = {
            "schema_version",
            "record_count",
            "records_per_shard",
            "modalities",
            "tokenizer_sha256",
            "tokenizer_vocab_size",
            "shards",
            "record_schema",
        }
        missing = required - set(payload)
        if missing:
            raise StoreSchemaError(f"store metadata is missing fields: {sorted(missing)}")
        unknown = set(payload) - required
        if unknown:
            raise StoreSchemaError(
                "store metadata contains fields unsupported by the current "
                f"schema: {sorted(unknown)}"
            )
        for key in (
            "schema_version",
            "record_count",
            "records_per_shard",
            "tokenizer_vocab_size",
        ):
            if not isinstance(payload[key], int) or isinstance(payload[key], bool):
                raise StoreSchemaError(f"store metadata {key} must be an integer")
        if isinstance(payload["modalities"], (str, bytes)) or not isinstance(
            payload["modalities"], Sequence
        ):
            raise StoreSchemaError("store metadata modalities must be an array")
        if isinstance(payload["shards"], (str, bytes)) or not isinstance(
            payload["shards"], Sequence
        ):
            raise StoreSchemaError("store metadata shards must be an array")
        if not isinstance(payload["tokenizer_sha256"], str):
            raise StoreSchemaError("store metadata tokenizer_sha256 must be a string")
        if not isinstance(payload["record_schema"], str):
            raise StoreSchemaError("store metadata record_schema must be a string")
        metadata = cls(
            schema_version=payload["schema_version"],
            record_count=payload["record_count"],
            records_per_shard=payload["records_per_shard"],
            modalities=tuple(payload["modalities"]),
            tokenizer_sha256=payload["tokenizer_sha256"],
            shards=tuple(payload["shards"]),
            tokenizer_vocab_size=payload["tokenizer_vocab_size"],
            record_schema=payload["record_schema"],
        )
        metadata.validate()
        return metadata


def write_store_metadata(store_dir: PathLike, metadata: StoreMetadata) -> Path:
    """Atomically write ``store.json``."""
    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)
    target = store_path / STORE_METADATA_NAME
    _atomic_write_json(target, metadata.to_dict())
    return target


def read_store_metadata(store_dir: PathLike) -> StoreMetadata:
    path = Path(store_dir) / STORE_METADATA_NAME
    if not path.is_file():
        raise StoreSchemaError(f"missing store metadata: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreSchemaError(f"failed to parse store metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise StoreSchemaError("the root of store.json must be an object")
    return StoreMetadata.from_dict(payload)


def recover_published_shard_sidecar(
    final_path: Path,
    sidecar_path: Path,
    *,
    shard_id: int,
    start_index: int,
    expected_records: int,
) -> Mapping[str, Any]:
    """Recover a missing sidecar from metadata embedded in a persisted LMDB.

    Recovery occurs only when the embedded range exactly matches the current
    writer contract, preventing an unknown directory from being mistaken for
    the current shard. Recovery rebuilds only the derivable sidecar and does
    not modify the LMDB data.
    """
    _require_lmdb()
    if not final_path.is_dir():
        raise StorageError(f"published shard is not a directory; cannot recover: {final_path}")
    data_file = final_path / "data.mdb"
    if not data_file.is_file():
        raise StorageError(f"published shard is missing data.mdb; cannot recover: {final_path}")
    environment = lmdb.open(
        str(final_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=True,
        max_readers=8,
    )
    try:
        with environment.begin(write=False, buffers=False) as transaction:
            raw_metadata = transaction.get(SHARD_METADATA_KEY)
    finally:
        environment.close()
    if raw_metadata is None:
        raise StorageError(
            f"published shard is missing embedded metadata; cannot recover: {final_path}"
        )
    try:
        embedded = json.loads(bytes(raw_metadata).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageError(f"published shard has corrupt embedded metadata: {final_path}") from exc
    expected = {
        "schema_version": STORE_SCHEMA_VERSION,
        "shard_id": int(shard_id),
        "start_index": int(start_index),
        "end_index_exclusive": int(start_index) + int(expected_records),
        "record_count": int(expected_records),
        "codec": "msgpack+zstd+sha256",
    }
    mismatches = {
        key: (embedded.get(key), value)
        for key, value in expected.items()
        if embedded.get(key) != value
    }
    if mismatches:
        raise StorageError(
            "published shard does not match the current writer contract; "
            f"recovery refused: {mismatches}"
        )
    recovered = {
        **expected,
        "sha256": _sha256_file(data_file),
        "data_file_bytes": data_file.stat().st_size,
    }
    _atomic_write_json(sidecar_path, recovered)
    return recovered


class LmdbShardWriter:
    """Single writer LMDB shard builder.

    The record index must increase continuously from ``start_index``. On normal context exit, all records are validated, committed, and the temporary directory is
    atomically published as ``shard-NNNNNN.lmdb``. On exceptional exit, only this instance's temporary directory is cleaned up; no published shards are modified.
    """

    def __init__(
        self,
        store_dir: PathLike,
        shard_id: int,
        start_index: int,
        expected_records: int,
        map_size: int,
        codec: Optional[RecordCodec] = None,
        commit_interval: int = 256,
    ) -> None:
        _require_lmdb()
        if shard_id < 0 or start_index < 0:
            raise ValueError("shard_id and start_index must be non-negative")
        if expected_records <= 0:
            raise ValueError("expected_records must be a positive integer")
        if map_size < 1024 * 1024:
            raise ValueError("map_size must be at least 1 MiB")
        if commit_interval <= 0:
            raise ValueError("commit_interval must be a positive integer")

        self.store_dir = Path(store_dir).resolve()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.shard_id = int(shard_id)
        self.start_index = int(start_index)
        self.expected_records = int(expected_records)
        self.map_size = int(map_size)
        self.codec = codec or RecordCodec()
        self.commit_interval = int(commit_interval)
        self.final_name = f"shard-{self.shard_id:06d}.lmdb"
        self.final_path = self.store_dir / self.final_name
        self.sidecar_path = self.store_dir / f"shard-{self.shard_id:06d}.json"
        if self.final_path.exists() and not self.sidecar_path.exists():
            recover_published_shard_sidecar(
                self.final_path,
                self.sidecar_path,
                shard_id=self.shard_id,
                start_index=self.start_index,
                expected_records=self.expected_records,
            )
            raise RecoveredShardError(
                f"Recovered shard sidecar; rerun resume: {self.sidecar_path}"
            )
        if self.final_path.exists() or self.sidecar_path.exists():
            raise FileExistsError(
                f"Shard already exists; refusing to overwrite: {self.final_path} / {self.sidecar_path}"
            )

        temporary = tempfile.mkdtemp(
            prefix=f".shard-{self.shard_id:06d}.tmp-",
            dir=str(self.store_dir),
        )
        self.temporary_path = Path(temporary).resolve()
        self._environment = lmdb.open(
            str(self.temporary_path),
            subdir=True,
            map_size=self.map_size,
            max_dbs=1,
            readonly=False,
            lock=True,
            sync=True,
            metasync=True,
            map_async=False,
            writemap=False,
            meminit=True,
        )
        self._transaction = self._environment.begin(write=True)
        self._count = 0
        self._next_index = self.start_index
        self._finalized = False

    @property
    def record_count(self) -> int:
        return self._count

    def _commit_batch(self) -> None:
        if self._transaction is None:
            raise StorageError("LMDB write transaction closed")
        self._transaction.commit()
        self._transaction = self._environment.begin(write=True)

    def put(self, record_index: int, record: Mapping[str, Any]) -> None:
        if self._finalized:
            raise StorageError("Sharding has been completed and writing cannot continue")
        if record_index != self._next_index:
            raise ValueError(
                f"expected record index {self._next_index}, got {record_index}"
            )
        if self._count >= self.expected_records:
            raise IncompleteShardError(
                f"Shard declares {self.expected_records} records; refusing to write an additional record"
            )
        payload = self.codec.encode(record)
        try:
            inserted = self._transaction.put(
                _record_key(record_index),
                payload,
                overwrite=False,
            )
        except lmdb.MapFullError as exc:
            raise StorageError(
                f"LMDB map_size={self.map_size} is insufficient; increase the shard map_size and rebuild"
            ) from exc
        if not inserted:
            raise StorageError(f"record_index={record_index} already exists")
        self._count += 1
        self._next_index += 1
        if self._count % self.commit_interval == 0:
            self._commit_batch()

    def _remove_temporary_path(self) -> None:
        if not self.temporary_path.exists():
            return
        if self.temporary_path.parent != self.store_dir:
            raise StorageError(
                f"Refusing to clean up a temporary directory outside store_dir: {self.temporary_path}"
            )
        shutil.rmtree(self.temporary_path)

    def abort(self) -> None:
        if self._finalized:
            return
        if self._transaction is not None:
            self._transaction.abort()
            self._transaction = None
        if self._environment is not None:
            self._environment.close()
            self._environment = None
        self._remove_temporary_path()

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise StorageError("Sharding has been finalized")
        if self._count != self.expected_records:
            count = self._count
            self.abort()
            raise IncompleteShardError(
                f"Shard actually wrote {count} records != expected {self.expected_records}"
            )
        if self._transaction is None or self._environment is None:
            raise StorageError("LMDB writer is down")

        shard_metadata = {
            "schema_version": STORE_SCHEMA_VERSION,
            "shard_id": self.shard_id,
            "start_index": self.start_index,
            "end_index_exclusive": self.start_index + self._count,
            "record_count": self._count,
            "codec": "msgpack+zstd+sha256",
        }
        encoded_metadata = json.dumps(
            shard_metadata,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        inserted = self._transaction.put(
            SHARD_METADATA_KEY,
            encoded_metadata,
            overwrite=False,
        )
        if not inserted:
            self.abort()
            raise StorageError("LMDB shard metadata key conflict")

        self._transaction.commit()
        self._transaction = None
        self._environment.sync(True)
        self._environment.close()
        self._environment = None

        data_file = self.temporary_path / "data.mdb"
        if not data_file.is_file():
            self._remove_temporary_path()
            raise StorageError(f"LMDB shard is missing data.mdb: {data_file}")
        shard_metadata["sha256"] = _sha256_file(data_file)
        shard_metadata["data_file_bytes"] = data_file.stat().st_size

        try:
            _fsync_directory(self.temporary_path)
            os.replace(self.temporary_path, self.final_path)
            _fsync_directory(self.store_dir)
            _atomic_write_json(self.sidecar_path, shard_metadata)
        except BaseException:
            if self.final_path.exists() and not self.sidecar_path.exists():
                raise StorageError(
                    f"Shard directory was published but sidecar write failed; inspect manually: {self.final_path}"
                )
            self._remove_temporary_path()
            raise

        self._finalized = True
        return shard_metadata

    def __enter__(self) -> "LmdbShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.abort()
            return False
        try:
            self.finalize()
        except BaseException:
            self.abort()
            raise
        return False


class _LmdbShardReader:
    def __init__(
        self,
        path: Path,
        codec: RecordCodec,
        readahead: bool,
    ) -> None:
        _require_lmdb()
        if not path.is_dir():
            raise FileNotFoundError(f"LMDB shard does not exist: {path}")
        self.path = path
        self.codec = codec
        self._environment = lmdb.open(
            str(path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=bool(readahead),
            meminit=True,
            max_readers=2048,
        )

    def get(self, record_index: int) -> Dict[str, Any]:
        with self._environment.begin(write=False, buffers=False) as transaction:
            payload = transaction.get(_record_key(record_index))
        if payload is None:
            raise KeyError(f"Shard {self.path.name} does not contain record_index={record_index}")
        return self.codec.decode(bytes(payload))

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()
            self._environment = None


class ShardedRecordStore:
    """Multi-process safe read-only sharded record view.

    LMDB handles are opened lazily per shard. During pickling, ``__getstate__`` closes existing handles. After a fork, ``_ensure_process_owner`` detects and
    closes inherited handles on the first reader access; each worker then reopens its read-only environment lazily.
    """

    def __init__(
        self,
        store_dir: PathLike,
        codec: Optional[RecordCodec] = None,
        readahead: bool = False,
        max_open_shards: int = 16,
    ) -> None:
        if not isinstance(readahead, bool):
            raise TypeError("readahead must be bool")
        if not isinstance(max_open_shards, int) or isinstance(
            max_open_shards,
            bool,
        ):
            raise TypeError("max_open_shards must be an integer")
        if max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer")
        self.store_dir = Path(store_dir).resolve()
        self.metadata = read_store_metadata(self.store_dir)
        self.codec = codec or RecordCodec()
        self.readahead = readahead
        self.max_open_shards = int(max_open_shards)
        self._owner_pid = os.getpid()
        self._readers: OrderedDict[int, _LmdbShardReader] = OrderedDict()
        for shard_id, shard_name in enumerate(self.metadata.shards):
            if Path(shard_name).name != shard_name:
                raise StoreSchemaError(f"shard name must not contain a directory: {shard_name}")
            expected_name = f"shard-{shard_id:06d}.lmdb"
            if shard_name != expected_name:
                raise StoreSchemaError(
                    f"shard[{shard_id}] name {shard_name!r} != "
                    f"{expected_name!r}"
                )
            path = self.store_dir / shard_name
            if not path.is_dir():
                raise StoreSchemaError(f"Shard referenced by store.json does not exist: {path}")
            sidecar_path = self.store_dir / f"shard-{shard_id:06d}.json"
            if not sidecar_path.is_file():
                raise StoreSchemaError(f"Missing shard sidecar: {sidecar_path}")
            try:
                sidecar = json.loads(
                    sidecar_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise StoreSchemaError(
                    f"Unable to parse shard sidecar: {sidecar_path}"
                ) from exc
            if not isinstance(sidecar, dict):
                raise StoreSchemaError(
                    f"shard sidecar root node must be an object: {sidecar_path}"
                )
            start_index = shard_id * self.metadata.records_per_shard
            expected_count = min(
                self.metadata.records_per_shard,
                self.metadata.record_count - start_index,
            )
            expected_sidecar = {
                "schema_version": STORE_SCHEMA_VERSION,
                "shard_id": shard_id,
                "start_index": start_index,
                "end_index_exclusive": start_index + expected_count,
                "record_count": expected_count,
                "codec": "msgpack+zstd+sha256",
            }
            expected_sidecar_fields = set(expected_sidecar) | {
                "sha256",
                "data_file_bytes",
            }
            if set(sidecar) != expected_sidecar_fields:
                raise StoreSchemaError(
                    f"shard sidecar field set does not match the current schema {sidecar_path}: "
                    f"missing={sorted(expected_sidecar_fields - set(sidecar))}, "
                    f"unknown={sorted(set(sidecar) - expected_sidecar_fields)}"
                )
            mismatches = {
                key: (sidecar.get(key), expected)
                for key, expected in expected_sidecar.items()
                if sidecar.get(key) != expected
            }
            if mismatches:
                raise StoreSchemaError(
                    f"shard sidecar range/schema mismatch {sidecar_path}: "
                    f"{mismatches}"
                )
            checksum = sidecar["sha256"]
            if not isinstance(checksum, str):
                raise StoreSchemaError(
                    f"shard sidecar SHA-256 must be a string: {sidecar_path}"
                )
            if len(checksum) != 64 or any(
                character not in "0123456789abcdef"
                for character in checksum
            ):
                raise StoreSchemaError(
                    f"shard sidecar is missing a valid SHA-256: {sidecar_path}"
                )
            data_file_bytes = sidecar.get("data_file_bytes")
            if (
                not isinstance(data_file_bytes, int)
                or isinstance(data_file_bytes, bool)
                or data_file_bytes <= 0
            ):
                raise StoreSchemaError(
                    f"shard sidecar is missing valid data_file_bytes: {sidecar_path}"
                )
            data_file = path / "data.mdb"
            if not data_file.is_file():
                raise StoreSchemaError(f"LMDB shard is missing data.mdb: {path}")
            actual_size = data_file.stat().st_size
            if actual_size != data_file_bytes:
                raise StoreSchemaError(
                    f"shard data.mdb size {actual_size} != sidecar "
                    f"{data_file_bytes}: {path}"
                )

    def __len__(self) -> int:
        return self.metadata.record_count

    def _ensure_process_owner(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._owner_pid:
            return
        for reader in self._readers.values():
            reader.close()
        self._readers = OrderedDict()
        self._owner_pid = current_pid

    def _reader(self, shard_id: int) -> _LmdbShardReader:
        self._ensure_process_owner()
        reader = self._readers.pop(shard_id, None)
        if reader is None:
            shard_path = self.store_dir / self.metadata.shards[shard_id]
            reader = _LmdbShardReader(shard_path, self.codec, self.readahead)
        self._readers[shard_id] = reader
        while len(self._readers) > self.max_open_shards:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def __getitem__(self, record_index: int) -> Dict[str, Any]:
        if not isinstance(record_index, (int, np.integer)):
            raise TypeError("record_index must be an integer")
        index = int(record_index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"record_index={record_index} is out of range, total records={len(self)}")
        shard_id = index // self.metadata.records_per_shard
        return self._reader(shard_id).get(index)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers = OrderedDict()

    def verify_shard_checksums(self) -> None:
        """Completely read data.mdb and check sidecar SHA-256 (used by the server for offline verification)."""
        for shard_id, shard_name in enumerate(self.metadata.shards):
            shard_path = self.store_dir / shard_name
            sidecar_path = self.store_dir / f"shard-{shard_id:06d}.json"
            if not sidecar_path.is_file():
                raise StoreSchemaError(f"Missing shard sidecar: {sidecar_path}")
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StoreSchemaError(f"Unable to parse shard sidecar: {sidecar_path}") from exc
            expected = str(sidecar.get("sha256", ""))
            actual = _sha256_file(shard_path / "data.mdb")
            if not expected or not hmac.compare_digest(expected, actual):
                raise CorruptRecordError(
                    f"Shard checksum verification failed for {shard_name}: expected={expected}, actual={actual}"
                )

    def __getstate__(self) -> Dict[str, Any]:
        self.close()
        state = dict(self.__dict__)
        state["_owner_pid"] = os.getpid()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Exceptions cannot be safely propagated during the destruction phase; explicit close/validation paths will still report all errors.
            return
