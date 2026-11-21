#!/usr/bin/env python3
"""Experimental, lossless Spin8-g128 storage inside a real GGUF v3 container.

This is a reference interchange format, NOT a llama.cpp inference backend.
The deliberately custom architecture makes stock model loaders reject it.
Only NumPy and the Python standard library are required. Large tensor data is
processed in bounded chunks; read_gguf returns read-only memory-mapped arrays.
"""
from __future__ import annotations

from atomic_output import publish_file

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

GGUF_VERSION = 3
ALIGNMENT = 32
SCHEMA = "org.spin8.g128.reference"
SCHEMA_VERSION = 1
CODEC = "spin8_g128_v1"
ARCHITECTURE = "spin8_qwen35"
GROUP_SIZE = 128
GROUP_BYTES = 34
RADIX = 3 ** 8
REGULAR_LIMIT = 9 * RADIX
FALLBACK_LIMIT = 65498  # 59049 + 6449 identity vectors (support != 2).
CHUNK_BYTES = 4 * 1024 * 1024
CHUNK_GROUPS = 65536
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 32 * 1024 * 1024
MAX_ITEMS = 1000000
MAX_TENSORS = 100000

# Values are from ggml/include/ggml.h and gguf-py/gguf/constants.py.
GGML_F32, GGML_F16, GGML_I8, GGML_F64 = 0, 1, 24, 28
PHYSICAL_DTYPES = {
    GGML_F32: np.dtype("<f4"), GGML_F16: np.dtype("<f2"),
    GGML_I8: np.dtype("i1"), GGML_F64: np.dtype("<f8"),
}
PHYSICAL_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_I8: "I8", GGML_F64: "F64"}
GROUP_DTYPE = np.dtype([("words", "<u2", (16,)), ("scale", "<f2")], align=False)
assert GROUP_DTYPE.itemsize == GROUP_BYTES

# Standard GGUF metadata types, including all legal primitive types for reads.
UINT8, INT8, UINT16, INT16, UINT32, INT32 = range(6)
FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(6, 13)
META_FORMATS = {
    UINT8: "B", INT8: "b", UINT16: "H", INT16: "h", UINT32: "I",
    INT32: "i", FLOAT32: "f", BOOL: "?", UINT64: "Q", INT64: "q", FLOAT64: "d",
}


class GGUFError(ValueError):
    """Malformed, incompatible, or corrupt Spin8 reference GGUF."""


def _align(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _string(value: str) -> bytes:
    if not isinstance(value, str):
        raise GGUFError("Expected a string")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_STRING_BYTES:
        raise GGUFError("String exceeds reference format limit")
    return struct.pack("<Q", len(encoded)) + encoded


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GGUFError("Metadata attributes must be finite JSON values") from error


def _name(value: Any, *, physical: bool = False) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise GGUFError("Tensor and metadata names must be nonempty strings without NUL")
    limit = 63 if physical else 65535
    if len(value.encode("utf-8")) > limit:
        raise GGUFError("Name is too long")
    return value


def _shape(value: Any, *, allow_scalar: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 4:
        raise GGUFError("Shape must contain at most four dimensions")
    if not value and not allow_scalar:
        raise GGUFError("Packed tensors need a nonempty shape")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
           or int(d) <= 0 or int(d) >= 2**63 for d in value):
        raise GGUFError("Shape dimensions must be positive signed 64-bit integers")
    shape = tuple(int(d) for d in value)
    if math.prod(shape) >= 2**63:
        raise GGUFError("Tensor element count overflows int64")
    return shape


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise GGUFError(f"{label} must be a boolean")
    return bool(value)


def _array_chunks(array: np.ndarray, dtype: np.dtype) -> Iterator[bytes]:
    """C-order bytes without materializing a whole noncontiguous tensor."""
    count = max(1, CHUNK_BYTES // dtype.itemsize)
    for start in range(0, array.size, count):
        part = np.asarray(array.flat[start:start + count], dtype=dtype)
        yield part.tobytes(order="C")


def _packed_chunks(words: np.ndarray, scales: np.ndarray) -> Iterator[bytes]:
    for start in range(0, len(scales), CHUNK_GROUPS):
        stop = min(start + CHUNK_GROUPS, len(scales))
        records = np.empty(stop - start, dtype=GROUP_DTYPE)
        records["words"] = words[start:stop]
        records["scale"] = scales[start:stop]
        yield records.tobytes()


def _digest(chunks: Iterator[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _validate_packed(packed: dict[str, Any], *, content: bool = True) -> None:
    if not isinstance(packed, dict):
        raise GGUFError("Packed tensors must be dictionaries")
    required = {"words", "scales", "bank", "shape", "group_size", "hadamard",
                "identity_fallback", "uniform_fallback"}
    if not required.issubset(packed):
        raise GGUFError(f"Missing packed fields: {sorted(required - packed.keys())}")
    group = packed["group_size"]
    if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)) or group != GROUP_SIZE:
        raise GGUFError("This schema supports group_size=128 only")
    shape = _shape(packed["shape"])
    count = math.prod(shape)
    if len(shape) != 2 or shape[1] % GROUP_SIZE:
        raise GGUFError("Packed weights must be [out,in] matrices with in divisible by 128")
    if packed.get("codec", CODEC) != CODEC:
        raise GGUFError("Unsupported packed codec version")
    words, scales, bank = (np.asarray(packed[key]) for key in ("words", "scales", "bank"))
    groups = count // GROUP_SIZE
    if words.dtype.kind != "u" or words.dtype.itemsize != 2 or words.shape != (groups, 16):
        raise GGUFError("words must be uint16 with shape [groups,16]")
    if scales.dtype.kind != "f" or scales.dtype.itemsize != 2 or scales.shape != (groups,):
        raise GGUFError("scales must be float16 with shape [groups]")
    if bank.dtype.kind != "f" or bank.dtype.itemsize != 8 or bank.shape != (9, 8, 8):
        raise GGUFError("bank must be float64 with shape [9,8,8]")
    hadamard = _bool(packed["hadamard"], "hadamard")
    identity = _bool(packed["identity_fallback"], "identity_fallback")
    uniform = _bool(packed["uniform_fallback"], "uniform_fallback")
    if not (hadamard and identity and uniform):
        raise GGUFError("spin8_g128_v1 requires H128 and both fallback modes")
    if not content:
        return
    if not np.isfinite(bank).all():
        raise GGUFError("Bank contains a nonfinite value")
    if (np.max(np.abs(bank @ bank.transpose(0, 2, 1) - np.eye(8))) > 1e-10
            or np.max(np.abs(np.linalg.det(bank) - 1.0)) > 1e-10):
        raise GGUFError("Bank must contain float64 SO(8) frames (tolerance 1e-10)")
    limit = FALLBACK_LIMIT if identity else REGULAR_LIMIT
    for start in range(0, groups, CHUNK_GROUPS):
        sc = scales[start:start + CHUNK_GROUPS]
        codes = words[start:start + CHUNK_GROUPS]
        if not np.isfinite(sc).all() or np.any(sc == 0):
            raise GGUFError("Scales must be finite and nonzero (including signed zero)")
        is_uniform = sc < 0
        if np.any(is_uniform) and not uniform:
            raise GGUFError("Negative scales require uniform_fallback=true")
        if np.any(codes[~is_uniform] >= limit):
            raise GGUFError("Reserved ternary codeword in a non-uniform group")


def _infer_meta(value: Any) -> tuple[int, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return BOOL, value
    if isinstance(value, str):
        return STRING, value
    if isinstance(value, int):
        if not -(2**63) <= value < 2**64:
            raise GGUFError("Metadata integer exceeds GGUF 64-bit range")
        return (INT64 if value < 0 else UINT64), value
    if isinstance(value, float) and math.isfinite(value):
        return FLOAT64, value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ITEMS:
            raise GGUFError("Metadata array is too long")
        if not value:
            return ARRAY, (STRING, [])
        pairs = [_infer_meta(item) for item in value]
        kinds = {kind for kind, _ in pairs}
        if kinds <= {UINT64, INT64}:
            numbers = [v for _, v in pairs]
            if min(numbers) < 0 and max(numbers) >= 2**63:
                raise GGUFError("Mixed integer array cannot fit signed int64")
            return ARRAY, (INT64 if min(numbers) < 0 else UINT64, numbers)
        if len(kinds) != 1 or ARRAY in kinds:
            raise GGUFError("GGUF arrays must contain one primitive metadata type")
        return ARRAY, (pairs[0][0], [v for _, v in pairs])
    raise GGUFError("Metadata needs finite scalar values or homogeneous scalar lists; serialize dictionaries as JSON strings")


def _meta_payload(kind: int, value: Any) -> bytes:
    if kind == STRING:
        return _string(value)
    if kind == ARRAY:
        subtype, items = value
        return struct.pack("<IQ", subtype, len(items)) + b"".join(_meta_payload(subtype, item) for item in items)
    try:
        return struct.pack("<" + META_FORMATS[kind], value)
    except (KeyError, struct.error) as error:
        raise GGUFError("Invalid metadata primitive") from error


def _base_metadata(tensor_map: dict[str, Any]) -> dict[str, tuple[int, Any]]:
    return {
        "general.architecture": (STRING, ARCHITECTURE),
        "general.alignment": (UINT32, ALIGNMENT),
        "spin8.schema": (STRING, SCHEMA),
        "spin8.schema_version": (UINT32, SCHEMA_VERSION),
        "spin8.codec": (STRING, CODEC),
        "spin8.group_size": (UINT32, GROUP_SIZE),
        "spin8.byte_order": (STRING, "little"),
        "spin8.hadamard_definition": (STRING, "normalized_sylvester_h128"),
        "spin8.payload_layout": (STRING, "16*u16_le+1*signed_f16_le"),
        "spin8.runtime_status": (STRING, "reference_storage_only"),
        "spin8.tensor_map": (STRING, _json(tensor_map)),
    }


def write_gguf(path: str | os.PathLike[str], tensors: dict[str, Any],
               metadata: dict[str, Any] | None = None, overwrite: bool = False) -> Path:
    """Atomically write a lossless experimental GGUF container.

    Packed dicts require all codec fields documented in SPEC.md. F16/F32
    ndarrays use ordinary GGUF tensors. Other fixed-size numeric ndarrays use
    I8 byte tensors plus explicit logical dtype/shape metadata. Object arrays
    and pickle are never accepted. Input arrays must remain unchanged during
    the two-pass hash/write operation. np.memmap inputs are supported.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if not isinstance(tensors, dict) or not tensors or len(tensors) > MAX_TENSORS // 2:
        raise GGUFError("Expected a nonempty, bounded dictionary of tensors")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise GGUFError("metadata must be a dictionary")
    tensor_map: dict[str, Any] = {}
    physical: list[dict[str, Any]] = []
    core_keys = {"words", "scales", "bank", "shape", "group_size", "hadamard",
                 "identity_fallback", "uniform_fallback"}
    for index, (name, value) in enumerate(tensors.items()):
        _name(name)
        prefix = f"spin8.t{index:06d}"
        if isinstance(value, dict):
            _validate_packed(value)
            words, scales, bank = (np.asarray(value[key]) for key in ("words", "scales", "bank"))
            shape = _shape(value["shape"])
            payload_name, bank_name = prefix + ".payload", prefix + ".bank"
            payload_iter = lambda w=words, s=scales: _packed_chunks(w, s)
            bank_iter = lambda b=bank: _array_chunks(b, np.dtype("<f8"))
            payload_hash, bank_hash = _digest(payload_iter()), _digest(bank_iter())
            attrs = {key: val for key, val in value.items() if key not in core_keys}
            if set(attrs) & {"kind", "payload", "groups", "payload_sha256", "bank_sha256", "attributes"}:
                raise GGUFError("Packed attributes must not use reserved tensor-map keys")
            attrs = json.loads(_json(attrs))
            tensor_map[name] = {
                "kind": "spin8_g128", "shape": list(shape), "payload": payload_name,
                "bank": bank_name, "groups": len(scales), "group_size": GROUP_SIZE,
                "hadamard": bool(value["hadamard"]),
                "identity_fallback": bool(value["identity_fallback"]),
                "uniform_fallback": bool(value["uniform_fallback"]),
                "payload_sha256": payload_hash, "bank_sha256": bank_hash,
                "attributes": attrs,
            }
            physical.extend([
                {"name": payload_name, "shape": (len(scales) * GROUP_BYTES,),
                 "type": GGML_I8, "nbytes": len(scales) * GROUP_BYTES,
                 "chunks": payload_iter, "sha256": payload_hash},
                {"name": bank_name, "shape": (9, 8, 8), "type": GGML_F64,
                 "nbytes": 9 * 8 * 8 * 8, "chunks": bank_iter, "sha256": bank_hash},
            ])
        else:
            if not isinstance(value, np.ndarray):
                raise GGUFError(f"{name}: expected a packed dictionary or numpy ndarray")
            array = np.asarray(value)
            shape = _shape(array.shape, allow_scalar=True)
            if array.size == 0 or array.dtype.kind not in "buif" or array.dtype.itemsize not in (1, 2, 4, 8):
                raise GGUFError("Only nonempty fixed-size numeric arrays are supported")
            dtype = array.dtype.newbyteorder("<")
            is_dense = dtype.kind == "f" and dtype.itemsize in (2, 4)
            ggml_type = (GGML_F16 if dtype.itemsize == 2 else GGML_F32) if is_dense else GGML_I8
            physical_name = prefix + (".dense" if is_dense else ".raw")
            chunks = lambda a=array, d=dtype: _array_chunks(a, d)
            digest = _digest(chunks())
            tensor_map[name] = {"kind": "dense" if is_dense else "raw_ndarray",
                                "shape": list(shape), "dtype": dtype.str,
                                "tensor": physical_name, "sha256": digest}
            physical.append({"name": physical_name, "shape": (shape or (1,)) if is_dense else (array.nbytes,),
                             "type": ggml_type, "nbytes": array.nbytes, "chunks": chunks, "sha256": digest})
    typed_meta = _base_metadata(tensor_map)
    for key, value in metadata.items():
        _name(key)
        if key.startswith("spin8.") or key in ("general.architecture", "general.alignment"):
            raise GGUFError(f"Reserved metadata key: {key}")
        typed_meta[key] = _infer_meta(value)
    if len(typed_meta) > MAX_ITEMS:
        raise GGUFError("Too many metadata entries")
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", GGUF_VERSION, len(physical), len(typed_meta)))
    for key, (kind, value) in typed_meta.items():
        header += _string(key) + struct.pack("<I", kind) + _meta_payload(kind, value)
        if len(header) > MAX_HEADER_BYTES:
            raise GGUFError("GGUF header exceeds reference limit")
    offset = 0
    for descriptor in physical:
        descriptor["offset"] = offset
        shape = descriptor["shape"]
        header += _string(_name(descriptor["name"], physical=True)) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *reversed(shape))
        header += struct.pack("<IQ", descriptor["type"], offset)
        offset = _align(offset + descriptor["nbytes"])
    if len(header) > MAX_HEADER_BYTES:
        raise GGUFError("GGUF header exceeds reference limit")
    header += b"\0" * (_align(len(header)) - len(header))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as output:
            temporary = output.name
            output.write(header)
            for descriptor in physical:
                digest, written = hashlib.sha256(), 0
                for chunk in descriptor["chunks"]():
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                if written != descriptor["nbytes"] or digest.hexdigest() != descriptor["sha256"]:
                    raise GGUFError("Input tensor changed while the file was being written")
                output.write(b"\0" * (_align(written) - written))
            output.flush()
            os.fsync(output.fileno())
        publish_file(temporary, path, overwrite=overwrite)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


class _Reader:
    def __init__(self, stream: BinaryIO, size: int):
        self.stream, self.size = stream, size

    def take(self, count: int) -> bytes:
        position = self.stream.tell()
        if count < 0 or position + count > min(self.size, MAX_HEADER_BYTES):
            raise GGUFError("Truncated or oversized GGUF header")
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFError("Truncated GGUF header")
        return data

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def string(self) -> str:
        length, = self.unpack("Q")
        if length > MAX_STRING_BYTES:
            raise GGUFError("String length exceeds limit")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GGUFError("Invalid UTF-8") from error

    def value(self, kind: int) -> Any:
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            subtype, count = self.unpack("IQ")
            if subtype == ARRAY or subtype not in (*META_FORMATS, STRING) or count > MAX_ITEMS:
                raise GGUFError("Invalid or oversized metadata array")
            return [self.value(subtype) for _ in range(count)]
        if kind == BOOL:
            raw, = self.unpack("B")
            if raw not in (0, 1):
                raise GGUFError("Invalid GGUF boolean")
            return bool(raw)
        if kind not in META_FORMATS:
            raise GGUFError("Unknown GGUF metadata type")
        value, = self.unpack(META_FORMATS[kind])
        if isinstance(value, float) and not math.isfinite(value):
            raise GGUFError("Nonfinite metadata is not allowed")
        return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GGUFError("Duplicate key in tensor-map JSON")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise GGUFError(f"Nonfinite JSON constant: {value}")


def _parse(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        reader = _Reader(stream, size)
        magic, version, count, kv_count = reader.unpack("4sIQQ")
        if magic != b"GGUF" or version != GGUF_VERSION:
            raise GGUFError("Expected little-endian GGUF version 3")
        if not 0 < count <= MAX_TENSORS or kv_count > MAX_ITEMS:
            raise GGUFError("Invalid GGUF entry counts")
        metadata, meta_types = {}, {}
        for _ in range(kv_count):
            key = _name(reader.string())
            if key in metadata:
                raise GGUFError("Duplicate GGUF metadata key")
            kind, = reader.unpack("I")
            metadata[key], meta_types[key] = reader.value(kind), kind
        required = _base_metadata({})
        if any(key.startswith("spin8.") and key not in required for key in metadata):
            raise GGUFError("Unrecognized reserved Spin8 schema field")
        for key, (kind, expected) in required.items():
            if key not in metadata or meta_types[key] != kind:
                raise GGUFError(f"Missing or wrongly typed schema field: {key}")
            if key != "spin8.tensor_map" and metadata[key] != expected:
                raise GGUFError(f"Unsupported schema field: {key}")
        physical: dict[str, Any] = {}
        for _ in range(count):
            name = _name(reader.string(), physical=True)
            if name in physical:
                raise GGUFError("Duplicate physical tensor name")
            ndim, = reader.unpack("I")
            if not 1 <= ndim <= 4:
                raise GGUFError("Physical tensor needs one to four dimensions")
            dimensions = reader.unpack("Q" * ndim)
            shape = _shape(tuple(reversed(dimensions)))
            ggml_type, offset = reader.unpack("IQ")
            if ggml_type not in PHYSICAL_DTYPES or offset % ALIGNMENT:
                raise GGUFError("Unsupported physical type or unaligned tensor offset")
            nbytes = math.prod(shape) * PHYSICAL_DTYPES[ggml_type].itemsize
            if nbytes > size:
                raise GGUFError("Physical tensor dimensions exceed file size")
            physical[name] = {"name": name, "shape": shape, "type": ggml_type,
                              "nbytes": nbytes, "offset": offset}
        table_end = stream.tell()
        data_start = _align(table_end)
        if data_start > size or any(reader.take(data_start - table_end)):
            raise GGUFError("Invalid header alignment padding")
        expected_offset = 0
        for descriptor in sorted(physical.values(), key=lambda item: item["offset"]):
            if descriptor["offset"] != expected_offset:
                raise GGUFError("Overlapping tensors or noncanonical tensor gaps")
            end = data_start + descriptor["offset"] + descriptor["nbytes"]
            if end > size:
                raise GGUFError("Tensor payload extends beyond file")
            expected_offset = _align(descriptor["offset"] + descriptor["nbytes"])
            pad_count = _align(descriptor["nbytes"]) - descriptor["nbytes"]
            stream.seek(end)
            if any(stream.read(pad_count)):
                raise GGUFError("Nonzero tensor alignment padding")
        if data_start + expected_offset != size:
            raise GGUFError("Truncated tensor padding or unexpected trailing bytes")
    try:
        tensor_map = json.loads(metadata["spin8.tensor_map"], object_pairs_hook=_unique_pairs,
                                parse_constant=_bad_constant)
    except (json.JSONDecodeError, RecursionError) as error:
        raise GGUFError("Invalid tensor-map JSON") from error
    if not isinstance(tensor_map, dict) or not tensor_map or len(tensor_map) > MAX_TENSORS:
        raise GGUFError("Invalid logical tensor map")
    return metadata, tensor_map, physical, data_start


def _valid_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise GGUFError("Invalid SHA-256 digest")
    return value


def _open_tensors(path: Path, *, verify_hashes: bool, verify_payload: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata, tensor_map, physical, data_start = _parse(path)
    tensors, used = {}, set()

    def lookup(name: Any) -> dict[str, Any]:
        if not isinstance(name, str) or name not in physical or name in used:
            raise GGUFError("Missing or multiply referenced physical tensor")
        used.add(name)
        return physical[name]

    def mmap(descriptor: dict[str, Any], dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=data_start + descriptor["offset"], shape=shape, order="C")

    def verify(descriptor: dict[str, Any], expected: Any) -> None:
        expected = _valid_hash(expected)
        if verify_hashes:
            mapped = mmap(descriptor, np.dtype("u1"), (descriptor["nbytes"],))
            if _digest(_array_chunks(mapped, np.dtype("u1"))) != expected:
                raise GGUFError(f"SHA-256 mismatch for {descriptor['name']}")

    for name, entry in tensor_map.items():
        _name(name)
        if not isinstance(entry, dict):
            raise GGUFError("Logical tensor entry must be an object")
        kind = entry.get("kind")
        shape = _shape(entry.get("shape"), allow_scalar=kind != "spin8_g128")
        if kind == "spin8_g128":
            needed = {"kind", "shape", "payload", "bank", "groups", "group_size", "hadamard",
                      "identity_fallback", "uniform_fallback", "payload_sha256", "bank_sha256", "attributes"}
            if set(entry) != needed:
                raise GGUFError("Invalid packed tensor schema fields")
            groups = math.prod(shape) // GROUP_SIZE
            if isinstance(entry["groups"], bool) or not isinstance(entry["groups"], int) or entry["groups"] != groups:
                raise GGUFError("Incorrect group count")
            payload, bank_descriptor = lookup(entry["payload"]), lookup(entry["bank"])
            if payload["type"] != GGML_I8 or payload["shape"] != (groups * GROUP_BYTES,):
                raise GGUFError("Packed payload must be a flat I8 tensor of 34 bytes per group")
            if bank_descriptor["type"] != GGML_F64 or bank_descriptor["shape"] != (9, 8, 8):
                raise GGUFError("Bank must be a [9,8,8] F64 tensor")
            attrs = entry["attributes"]
            if not isinstance(attrs, dict) or any(key in needed or key in ("words", "scales") for key in attrs):
                raise GGUFError("Attributes must not override codec fields")
            records = mmap(payload, GROUP_DTYPE, (groups,))
            packed = dict(attrs)
            packed.update({"words": records["words"], "scales": records["scale"],
                           "bank": mmap(bank_descriptor, np.dtype("<f8"), (9, 8, 8)),
                           "shape": shape, "group_size": entry["group_size"],
                           "hadamard": entry["hadamard"], "identity_fallback": entry["identity_fallback"],
                           "uniform_fallback": entry["uniform_fallback"]})
            _validate_packed(packed, content=verify_payload)
            verify(payload, entry["payload_sha256"])
            verify(bank_descriptor, entry["bank_sha256"])
            tensors[name] = packed
        elif kind in ("dense", "raw_ndarray"):
            if set(entry) != {"kind", "shape", "dtype", "tensor", "sha256"}:
                raise GGUFError("Invalid dense/raw tensor schema fields")
            if not isinstance(entry["dtype"], str) or len(entry["dtype"]) > 64:
                raise GGUFError("Logical dtype must be a short NumPy dtype string")
            try:
                dtype = np.dtype(entry["dtype"])
            except (TypeError, ValueError) as error:
                raise GGUFError("Invalid logical dtype") from error
            if dtype.kind not in "buif" or dtype.itemsize not in (1, 2, 4, 8) or dtype.str != dtype.newbyteorder("<").str:
                raise GGUFError("Only little-endian fixed-size numeric dtypes are supported")
            descriptor = lookup(entry["tensor"])
            nbytes = math.prod(shape) * dtype.itemsize
            if kind == "dense":
                allowed = GGML_F16 if dtype == np.dtype("<f2") else GGML_F32 if dtype == np.dtype("<f4") else None
                if descriptor["type"] != allowed or descriptor["shape"] != (shape or (1,)):
                    raise GGUFError("Dense tensor type or shape mismatch")
            elif descriptor["type"] != GGML_I8 or descriptor["shape"] != (nbytes,):
                raise GGUFError("Raw array must be a flat I8 tensor")
            if descriptor["nbytes"] != nbytes:
                raise GGUFError("Logical tensor byte count mismatch")
            tensors[name] = mmap(descriptor, dtype, shape)
            verify(descriptor, entry["sha256"])
        else:
            raise GGUFError("Unsupported logical tensor kind")
    if used != set(physical):
        raise GGUFError("Unmapped physical tensor")
    return tensors, metadata, {"tensor_map": tensor_map, "physical": physical, "data_offset": data_start}


def read_gguf(path: str | os.PathLike[str], *, verify_hashes: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (logical tensors, all GGUF metadata), validating codec contents.

    Arrays are read-only views into memory-mapped file storage; keep the file
    unchanged while they are in use. Hash checks make one streaming pass over
    all tensor bytes by default. No pickle or model code is executed.
    """
    tensors, metadata, _ = _open_tensors(Path(path), verify_hashes=verify_hashes, verify_payload=True)
    return tensors, metadata


def inspect_gguf(path: str | os.PathLike[str], *, verify: bool = False) -> dict[str, Any]:
    """Inspect structure cheaply; verify=True additionally scans hashes/codes."""
    path = Path(path)
    _, metadata, info = _open_tensors(path, verify_hashes=verify, verify_payload=verify)
    logical = {}
    for name, entry in info["tensor_map"].items():
        logical[name] = {key: value for key, value in entry.items() if key not in ("attributes",)}
        if entry["kind"] == "spin8_g128":
            logical[name]["payload_bytes"] = entry["groups"] * GROUP_BYTES
            logical[name]["bank_bytes"] = 4608
            logical[name]["payload_bits_per_weight"] = GROUP_BYTES * 8 / GROUP_SIZE
    return {"path": str(path), "gguf_version": GGUF_VERSION, "architecture": ARCHITECTURE,
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "runtime_status": "reference_storage_only",
            "file_bytes": path.stat().st_size, "data_offset": info["data_offset"],
            "hashes_and_codec_validated": verify,
            "logical_tensors": logical,
            "physical_tensors": [{**entry, "type_name": PHYSICAL_NAMES[entry["type"]]} for entry in info["physical"].values()],
            "metadata": {key: value for key, value in metadata.items() if key != "spin8.tensor_map"}}


def validate_gguf(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully validate structural bounds, codec invariants, and SHA-256 hashes."""
    return inspect_gguf(path, verify=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read the schema and tensor table")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--verify", action="store_true", help="Also scan all hashes and codec contents")
    validate = sub.add_parser("validate", help="Fully validate the reference container")
    validate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_gguf(args.path, verify=args.command == "validate" or args.verify)
    except (GGUFError, OSError) as error:
        parser.exit(1, f"Invalid Spin8 GGUF: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
