"""Validated, single-writer tensor checkpoints for a Spin8 encoding run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import reference_gguf as reference

SCHEMA = "spin8-native-tensor-cache-v1"
BANK_BYTES = 9 * 8 * 8 * 8


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path, limit: int) -> Any:
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Invalid cache record: " + str(path))
    with path.open("rb") as source:
        return json.loads(source.read(limit + 1))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink():
        raise ValueError("Cache partial file must not be a symlink")
    with partial.open("wb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, path)


class TensorCache:
    """One directory belongs to one immutable run configuration and one writer."""

    def __init__(self, root: Path, settings: dict[str, Any]):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = json.loads(_json(settings))
        self.fingerprint = hashlib.sha256(_json(self.settings)).hexdigest()
        self.stats = {"hits": 0, "written": 0, "invalidated": 0}
        manifest = self.root / "run.json"
        expected = {"schema": SCHEMA, "fingerprint": self.fingerprint, "settings": self.settings}
        if manifest.exists() or manifest.is_symlink():
            try:
                saved = _read_json(manifest, 4 << 20)
            except (ValueError, OSError) as error:
                raise ValueError("Unreadable resume manifest; preserve it and choose another --resume-dir") from error
            if saved != expected:
                raise ValueError("Resume settings/source/code mismatch; use the original settings or a new --resume-dir")
        else:
            remaining = [p for p in self.root.iterdir() if p.name != "run.json.partial"]
            if remaining:
                raise ValueError("Resume directory has files but no run.json; choose an empty directory")
            _atomic_bytes(manifest, _json(expected))

    def _paths(self, name: str) -> tuple[Path, Path, Path]:
        stem = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return tuple(self.root / (stem + suffix) for suffix in (".payload", ".bank.f64", ".json"))

    def _invalidate(self, paths: tuple[Path, Path, Path]) -> None:
        for path in paths:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".partial").unlink(missing_ok=True)
        self.stats["invalidated"] += 1

    def load(self, name: str, shape: tuple[int, ...], *, count_hit: bool = True) -> tuple[dict[str, Any], np.memmap] | None:
        shape = tuple(int(n) for n in shape)
        if len(shape) != 2 or min(shape) <= 0 or shape[1] % 128:
            raise ValueError("Cache requires a positive 128-aligned matrix shape")
        paths = self._paths(name)
        payload_path, bank_path, record_path = paths
        if not any(p.exists() or p.is_symlink() or p.with_name(p.name + ".partial").exists() for p in paths):
            return None
        try:
            record = _read_json(record_path, 65536)
            payload_bytes = shape[0] * (shape[1] // 128) * 34
            expected = {"schema": SCHEMA, "fingerprint": self.fingerprint, "tensor": name,
                        "shape": list(shape), "payload_bytes": payload_bytes, "bank_bytes": BANK_BYTES}
            if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
                raise ValueError("Cache tensor record mismatch")
            for path, size, key in ((payload_path, payload_bytes, "payload_sha256"),
                                    (bank_path, BANK_BYTES, "bank_sha256")):
                if path.is_symlink() or path.stat().st_size != size or _hash(path) != record.get(key):
                    raise ValueError("Cache tensor size/hash mismatch")
            payload = np.memmap(payload_path, mode="r", dtype="u1", shape=(shape[0], shape[1] // 128 * 34))
            rows = payload.reshape(-1).view(reference.GROUP_DTYPE)
            bank = np.memmap(bank_path, mode="r", dtype="<f8", shape=(9, 8, 8))
            packed = {"codec": "spin8_g128_v1", "words": rows["words"], "scales": rows["scale"],
                      "bank": bank, "shape": shape, "group_size": 128, "hadamard": True,
                      "identity_fallback": True, "uniform_fallback": True}
            reference._validate_packed(packed)
        except (ValueError, OSError, KeyError, TypeError) as error:
            print("Retrying invalid/incomplete Spin8 cache tensor " + name + ": " + str(error), flush=True)
            self._invalidate(paths)
            return None
        if count_hit:
            self.stats["hits"] += 1
        return packed, payload

    def store(self, name: str, packed: dict[str, Any]) -> tuple[dict[str, Any], np.memmap]:
        reference._validate_packed(packed)
        shape = tuple(int(n) for n in packed["shape"])
        payload_path, bank_path, record_path = self._paths(name)
        partial = payload_path.with_name(payload_path.name + ".partial")
        if partial.is_symlink():
            raise ValueError("Cache partial file must not be a symlink")
        digest = hashlib.sha256()
        with partial.open("wb") as target:
            for start in range(0, len(packed["scales"]), 65536):
                stop = min(start + 65536, len(packed["scales"]))
                records = np.empty(stop - start, dtype=reference.GROUP_DTYPE)
                records["words"] = packed["words"][start:stop]
                records["scale"] = packed["scales"][start:stop]
                data = records.tobytes()
                target.write(data)
                digest.update(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(partial, payload_path)
        bank_data = np.asarray(packed["bank"], dtype="<f8").tobytes(order="C")
        _atomic_bytes(bank_path, bank_data)
        record = {"schema": SCHEMA, "fingerprint": self.fingerprint, "tensor": name,
                  "shape": list(shape), "payload_bytes": shape[0] * shape[1] // 128 * 34,
                  "bank_bytes": BANK_BYTES, "payload_sha256": digest.hexdigest(),
                  "bank_sha256": hashlib.sha256(bank_data).hexdigest()}
        _atomic_bytes(record_path, _json(record))
        self.stats["written"] += 1
        loaded = self.load(name, shape, count_hit=False)
        if loaded is None:
            raise ValueError("Newly written tensor cache failed validation: " + name)
        return loaded
