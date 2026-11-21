#!/usr/bin/env python3
"""Convert a local HF Safetensors snapshot directly to the private Spin8 GGUF.

The pinned llama.cpp converter owns all name/layout/tokenizer transformations.
This adapter captures its lazy canonical tensors and replaces the final float
write with tiled weight edits and Spin8 encoding. It never stages a complete
F16/F32 GGUF. Large plain cast chains are sliced at the Safetensors mmap before
casting; other transformations are evaluated one canonical tensor at a time.

This is an experimental codec, not a standard llama.cpp GGUF quantization type.
No answer data or optimizer is used and no capability improvement is implied.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any

import numpy as np


BASE_COMMIT = "7fce5439fe451cd48340b112ec37d2befe02c776"
SCHEMA = "spin8-hf-direct-v1"
SUPPORTED_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5TextModel",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_modules(root: Path) -> dict[str, Any]:
    root = root.resolve()
    for relative in ("", "gguf-py", "tools/spin8"):
        sys.path.insert(0, str(root / relative))
    modules = {name: importlib.import_module(name) for name in (
        "gguf", "conversion", "convert", "reference_codec", "reference_gguf",
        "resume", "edit_weights", "atomic_output", "torch",
    )}
    for name in ("gguf", "conversion", "convert", "reference_codec"):
        if not Path(modules[name].__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"{name} was imported from another checkout; use a fresh subprocess")
    gguf = modules["gguf"]
    if int(gguf.GGMLQuantizationType.SPIN8_G128) != 44:
        raise RuntimeError("The checkout does not contain the expected private Spin8 ABI")
    return modules


def _snapshot_identity(directory: Path, revision: str | None) -> dict[str, Any]:
    """Hash actual local inputs, including the index and external chat templates.

    Hashes are deliberately not trusted solely from mtimes or download metadata:
    resuming with altered weights must never reuse a stale tensor checkpoint.
    """
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError(f"Expected a downloaded HF snapshot with config.json: {directory}")
    manifest_path = directory / "source.json"
    manifest, expected = None, {}
    if manifest_path.exists():
        if manifest_path.stat().st_size > 16 << 20:
            raise ValueError("HF source.json manifest is unexpectedly large")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "spin8-hf-source-v2" or manifest.get("status") != "COMPLETE"
                or manifest.get("weight_payload_downloaded") is not True):
            raise ValueError("HF source.json must be a recognized COMPLETE download manifest")
        manifest_revision = manifest.get("revision", "")
        if not isinstance(manifest_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", manifest_revision):
            raise ValueError("HF source.json must pin a full commit SHA")
        if revision is not None and revision != manifest_revision:
            raise ValueError("Requested source revision differs from source.json")
        revision = manifest_revision
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("HF source.json contains no verified files")
        for entry in entries:
            name = entry.get("path", "") if isinstance(entry, dict) else ""
            if (not isinstance(name, str) or not name or name in expected or Path(name).is_absolute()
                    or ".." in Path(name).parts or name == "source.json" or entry.get("verified") is not True
                    or type(entry.get("size")) is not int or entry["size"] < 0
                    or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
                raise ValueError("HF source.json contains an invalid/unverified file record")
            expected[name] = entry
        core = {"repo": manifest.get("repo"), "revision": revision,
                "files": [{"path": name, "size": entry["size"], "sha256": entry["sha256"]}
                          for name, entry in sorted(expected.items())]}
        fingerprint = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        if fingerprint != manifest.get("content_fingerprint"):
            raise ValueError("HF source.json content fingerprint differs from its records")
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and not any(part.startswith(".") for part in path.relative_to(directory).parts)
                   and path.name != "source.json"
                   and (path.relative_to(directory).as_posix() in expected
                        or path.suffix in {".safetensors", ".json", ".jinja", ".model", ".txt"}
                        or path.name == "README.md"))
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Direct streaming requires local Safetensors weights")
    if manifest is not None and {path.relative_to(directory).as_posix() for path in files} != set(expected):
        raise ValueError("HF source files differ from the verified source.json inventory")
    records = []
    for path in files:
        stat = path.stat()
        if path.suffix == ".safetensors":
            print(f"Hashing source shard {path.name} ({stat.st_size:,} bytes)", flush=True)
        digest = _sha256(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while hashing: {path}")
        name = path.relative_to(directory).as_posix()
        if manifest is not None:
            if path.is_symlink() or stat.st_size != expected[name]["size"] or digest != expected[name]["sha256"]:
                raise ValueError(f"Actual HF file hash/size differs from source.json: {name}")
        records.append({"file": name, "bytes": stat.st_size, "sha256": digest})
    return {"revision": revision, "files": records,
            "repo": manifest.get("repo") if manifest else None,
            "verified_source_manifest": manifest is not None,
            "content_fingerprint": manifest.get("content_fingerprint") if manifest else None}


def _plain_local_leaf(value: Any, modules: dict[str, Any]) -> Any | None:
    """Recognize only the pinned converter's pointwise dtype/numpy cast chain.

    Never push slices through permutations, concatenations, arithmetic, or a
    generic callable merely because the output happens to have the same shape.
    """
    gguf, torch = modules["gguf"], modules["torch"]
    shape = tuple(value.shape)
    while isinstance(value, gguf.LazyBase) and value._data is None:
        function = value._func
        qualname = getattr(function, "__qualname__", "")
        if qualname == "LazyTorchTensor.from_local_tensor.<locals>.<lambda>":
            if len(value._args) == 1 and isinstance(value._args[0], gguf.utility.LocalTensor):
                leaf = value._args[0]
                return leaf if tuple(leaf.shape) == shape and leaf.dtype in {"BF16", "F16", "F32"} else None
            return None
        permitted = function is torch.Tensor.contiguous
        if function is torch.Tensor.to:
            permitted = len(value._args) == 2 and value._args[1] is torch.float32 and not value._kwargs
        if qualname == "LazyMeta.__new__.<locals>.__getattr__.<locals>.<lambda>":
            closure = dict(zip(function.__code__.co_freevars,
                               (cell.cell_contents for cell in function.__closure__ or ())))
            permitted = (closure.get("name") == "to" and len(value._args) == 2
                         and value._args[1] is torch.float32 and not value._kwargs)
        permitted |= qualname in {
            "LazyTorchTensor.numpy.<locals>.<lambda>",
            "LazyNumpyTensor.astype.<locals>.<lambda>",
        }
        if not permitted or not value._args or not isinstance(value._args[0], gguf.LazyBase):
            return None
        if any(isinstance(arg, gguf.LazyBase) for arg in value._args[1:]):
            return None
        value = value._args[0]
        if tuple(value.shape) != shape:
            return None
    return None


@dataclass
class CanonicalTensor:
    name: str
    data: Any
    shape: tuple[int, ...]
    dtype: np.dtype
    qtype: Any
    leaf: Any | None = None
    eager: np.ndarray | None = None

    def read(self, selection: tuple[slice, ...], modules: dict[str, Any],
             max_materialized_bytes: int, stats: dict[str, Any]) -> np.ndarray:
        if self.leaf is not None:
            raw = self.leaf.mmap_bytes()
            dtype = "<u2" if self.leaf.dtype == "BF16" else ("<f2" if self.leaf.dtype == "F16" else "<f4")
            values = raw.view(dtype).reshape(self.shape)[selection]
            if self.leaf.dtype == "BF16":
                values = (values.astype("<u4") << 16).view("<f4")
            return np.array(values, dtype=self.dtype, copy=True)
        if self.eager is None:
            size = math.prod(self.shape) * self.dtype.itemsize
            if size > max_materialized_bytes:
                raise MemoryError(f"Canonical transform {self.name} requires {size:,} result bytes; "
                                  f"limit is {max_materialized_bytes:,}. Increase the explicit limit only "
                                  "with sufficient host RAM; plain embeddings are streamed separately.")
            self.eager = modules["gguf"].LazyBase.to_eager(self.data)
            if not isinstance(self.eager, np.ndarray) or self.eager.shape != self.shape:
                raise ValueError(f"Unexpected canonical array for {self.name}")
            stats["max_materialized_canonical_tensor_bytes"] = max(
                stats["max_materialized_canonical_tensor_bytes"], self.eager.nbytes)
        return np.array(self.eager[selection], dtype=self.dtype, copy=True)

    def release(self) -> None:
        self.data = self.eager = self.leaf = None


def _capture(directory: Path, output: Path, modules: dict[str, Any], *, no_mtp: bool) -> tuple[Any, dict[str, CanonicalTensor]]:
    conversion, gguf = modules["conversion"], modules["gguf"]
    hparams = conversion.ModelBase.load_hparams(directory, False)
    architecture = conversion.get_model_architecture(hparams, conversion.ModelType.TEXT)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"HF architecture {architecture!r} is not supported by the qwen35 Spin8 runtime")
    text_config = {**hparams, **hparams.get("text_config", {})}
    if hparams.get("quantization_config") or text_config.get("quantization_config"):
        raise ValueError("Start from full floating weights, not an already quantized HF checkpoint")
    original_class = conversion.get_model_class(architecture)
    # The upstream MTP mixin keeps class state while indexing. Isolate every
    # snapshot and do not mutate the shared registry when loading a donor.
    model_class = type("Spin8SnapshotModel", (original_class,), {
        "model_arch": original_class.model_arch, "no_mtp": no_mtp,
        "mtp_only": False, "_original_block_count": None, "opt_num_mtp_layers": 0,
    })
    model = model_class(directory, gguf.LlamaFileType.MOSTLY_F16, output,
                        eager=False, hparams=hparams)
    if model.hadamard_folded_names():
        raise ValueError("Folded Hadamard checkpoints require a different coordinate system")
    tensors: dict[str, CanonicalTensor] = {}
    writer = model.gguf_writer

    def add_tensor(name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        if raw_shape is not None or tensor_endianess is not None:
            raise ValueError("Unexpected raw shape/endianness from canonical HF conversion")
        if raw_dtype not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            raise ValueError(f"Expected canonical F16/F32 tensor, got {name}: {raw_dtype}")
        if isinstance(tensor, gguf.LazyChunkedTensor):
            raise ValueError("This architecture supplied a chunked tensor outside the supported dense Qwen path")
        shape = tuple(int(n) for n in tensor.shape)
        if not shape or any(n <= 0 for n in shape):
            raise ValueError(f"Invalid tensor dimensions: {name} {shape}")
        if name in tensors:
            raise ValueError(f"Duplicate canonical tensor: {name}")
        spec = CanonicalTensor(name, tensor, shape, np.dtype(tensor.dtype), raw_dtype)
        spec.leaf = _plain_local_leaf(tensor, modules)
        tensors[name] = spec
        writer.add_tensor_info(name, shape, spec.dtype, math.prod(shape) * spec.dtype.itemsize,
                               raw_dtype=raw_dtype)

    writer.add_tensor = add_tensor
    model.prepare_tensors()
    model.prepare_metadata(vocab_only=False)
    if not tensors:
        raise ValueError("HF checkpoint yielded no canonical text tensors")
    return model, tensors


def _validate_donor(source: Any, donor: Any, source_tensors: dict[str, CanonicalTensor],
                    donor_tensors: dict[str, CanonicalTensor]) -> None:
    a, b = source.gguf_writer.kv_data[0], donor.gguf_writer.kv_data[0]
    keys = {key for key in a.keys() | b.keys() if key.startswith(("qwen35.", "tokenizer."))}
    different = [key for key in sorted(keys) if key not in a or key not in b or a[key] != b[key]]
    if different:
        raise ValueError("Math donor metadata/tokenizer differs: " + ", ".join(different[:12]))
    if source_tensors.keys() != donor_tensors.keys():
        raise ValueError("Math donor canonical tensor inventory differs from the source")
    for name, value in source_tensors.items():
        if value.shape != donor_tensors[name].shape or value.qtype != donor_tensors[name].qtype:
            raise ValueError(f"Math donor canonical tensor shape/type differs: {name}")


def _matrix_kind(spec: CanonicalTensor, model: Any, modules: dict[str, Any]) -> Any | None:
    kind = modules["convert"].canonical_weight_kind(spec.name, model.tensor_map)
    if len(spec.shape) == 2 and kind in modules["convert"].MATRIX_KINDS:
        return kind
    return None


def _slices(shape: tuple[int, ...], tile_elements: int, axis: str | None, quantized: bool):
    if len(shape) != 2:
        yield tuple(slice(None) for _ in shape)
        return
    tile_axis = 1 if axis == "left" else 0
    step = max(1, tile_elements // shape[1 - tile_axis])
    if quantized and tile_axis == 1:
        step = max(128, step // 128 * 128)
    for start in range(0, shape[tile_axis], step):
        selection = [slice(None), slice(None)]
        selection[tile_axis] = slice(start, min(start + step, shape[tile_axis]))
        yield tuple(selection)


def convert_hf_to_spin8(source_dir: str | Path, output_path: str | Path, *,
                       llama_cpp_dir: str | Path, cache_dir: str | Path,
                       encoder: str = "torch", device: str = "cuda", compute_dtype: str = "float32",
                       donor_dir: str | Path | None = None, alpha: float = 0.0,
                       direction_path: str | Path | None = None, math_include: list[str] | None = None,
                       source_revision: str | None = None, donor_revision: str | None = None,
                       include_embedding: bool = True, bank_kind: str = "triality", seed: int = 81237,
                       scale_steps: int = 17, chunk_groups: int = 256,
                       tile_elements: int = 1 << 20, max_materialized_bytes: int = 2 << 30,
                       no_mtp: bool = True, force: bool = False) -> dict[str, Any]:
    """Return a measured conversion report; inference and quality remain false.

    Optional math editing is same-base interpolation W + alpha*(donor-W), in
    canonical coordinates. A supplied direction is removed after interpolation.
    Its semantic meaning cannot be inferred from the vector's numeric contents.
    """
    source_dir, output_path = Path(source_dir).resolve(), Path(output_path).resolve()
    llama_cpp_dir, cache_dir = Path(llama_cpp_dir).resolve(), Path(cache_dir).resolve()
    donor_dir = Path(donor_dir).resolve() if donor_dir is not None else None
    direction_path = Path(direction_path).resolve() if direction_path is not None else None
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if alpha and donor_dir is None:
        raise ValueError("A nonzero math alpha requires a same-base math donor snapshot")
    if type(tile_elements) is not int or tile_elements < 128:
        raise ValueError("tile_elements must be an integer >= 128")
    if type(max_materialized_bytes) is not int or max_materialized_bytes < 128:
        raise ValueError("max_materialized_bytes must be an integer >= 128")
    if encoder not in {"numpy", "torch"} or compute_dtype not in {"float32", "float64"}:
        raise ValueError("Unsupported encoder or compute dtype")
    if output_path.exists() and not force:
        raise FileExistsError(output_path)
    for directory in (source_dir, donor_dir):
        if directory is not None and (output_path.is_relative_to(directory) or cache_dir.is_relative_to(directory)):
            raise ValueError("Keep output and cache outside the immutable HF source snapshots")
    if output_path.is_relative_to(cache_dir):
        raise ValueError("Keep final GGUF outside the tensor checkpoint directory")
    modules = _load_modules(llama_cpp_dir)
    gguf, torch, codec = modules["gguf"], modules["torch"], modules["reference_codec"]
    if encoder == "numpy" and device != "cpu":
        raise ValueError("The NumPy encoder requires device='cpu'")
    torch_device = torch.device(device)
    if encoder == "torch" and torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this process has no available CUDA device")
        # Execute an actual CUDA operation before touching a large checkpoint.
        probe = torch.arange(8, device=torch_device, dtype=torch.float32)
        if float((probe * probe).sum().item()) != 140.0:
            raise RuntimeError("CUDA arithmetic preflight failed")
        torch.cuda.synchronize(torch_device)
    elif torch_device.type != "cpu":
        raise ValueError("Only CPU and CUDA encoder devices are supported")
    device = str(torch_device)
    encode = codec.encode if encoder == "numpy" else importlib.import_module("reference_torch").encode_torch
    bank = codec.make_bank(bank_kind, seed)
    include = [re.compile(pattern) for pattern in (math_include or [])]
    source_identity = _snapshot_identity(source_dir, source_revision)
    donor_identity = _snapshot_identity(donor_dir, donor_revision) if donor_dir is not None else None
    source_revision = source_identity["revision"]
    donor_revision = donor_identity["revision"] if donor_identity is not None else None
    source, source_tensors = _capture(source_dir, output_path, modules, no_mtp=no_mtp)
    donor, donor_tensors = None, {}
    if donor_dir is not None:
        donor, donor_tensors = _capture(donor_dir, output_path, modules, no_mtp=no_mtp)
        _validate_donor(source, donor, source_tensors, donor_tensors)
    width = int(source.hparams["hidden_size"])
    direction = modules["edit_weights"].normalize_direction(direction_path, width) if direction_path else None
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
    )}
    code_paths = [Path(__file__), *(llama_cpp_dir / name for name in (
        "conversion/base.py", "conversion/qwen.py", "conversion/__init__.py",
        "gguf-py/gguf/lazy.py", "gguf-py/gguf/utility.py", "gguf-py/gguf/quants.py",
        "gguf-py/gguf/gguf_writer.py", "gguf-py/gguf/constants.py", "gguf-py/gguf/tensor_mapping.py",
        "tools/spin8/reference_codec.py", "tools/spin8/reference_torch.py", "tools/spin8/triality_lie.py",
        "tools/spin8/reference_gguf.py", "tools/spin8/resume.py", "tools/spin8/edit_weights.py",
        "tools/spin8/convert.py", "tools/spin8/atomic_output.py",
    ))]
    settings = {"schema": SCHEMA, "base_commit": BASE_COMMIT, "source": source_identity,
                "donor": donor_identity, "alpha": alpha, "math_include": math_include or [],
                "direction_sha256": _sha256(direction_path) if direction_path else None,
                "encoder": encoder, "device": device, "compute_dtype": compute_dtype,
                "include_embedding": include_embedding, "no_mtp": no_mtp, "seed": seed,
                "bank_kind": bank_kind, "scale_steps": scale_steps, "chunk_groups": chunk_groups,
                "tile_elements": tile_elements, "versions": versions, "python": platform.python_version(),
                "code": {str(path.relative_to(llama_cpp_dir)) if path.is_relative_to(llama_cpp_dir)
                         else "hf_stream.py": _sha256(path) for path in code_paths},
                "torch_cuda": torch.version.cuda, "matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "gpu_name": torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None,
                "inventory": [{"name": name, "shape": spec.shape, "dtype": str(spec.dtype)}
                              for name, spec in source_tensors.items()]}
    cache = modules["resume"].TensorCache(cache_dir, settings)
    stats = {"max_materialized_canonical_tensor_bytes": 0, "max_edit_tile_bytes": 0,
             "encoded_tensors_this_run": 0,
             "plain_mmap_tensor_count": sum(spec.leaf is not None for spec in source_tensors.values()),
             "canonical_tensor_count": len(source_tensors)}
    records, left_names, right_names, edited_names, selected_names = [], [], [], [], []
    writer = source.gguf_writer
    # All parameter counts/metadata have already been computed using canonical
    # logical float shapes, not companion-bank or packed byte dimensions.
    writer.tensors = [{}]
    writer.remove_key("general.file_type")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spin8-hf-work-", dir=output_path.parent) as directory:
        work = Path(directory)
        descriptors = []
        for index, name in enumerate(list(source_tensors)):
            spec = source_tensors.pop(name)
            donor_spec = donor_tensors.pop(name, None)
            kind = _matrix_kind(spec, source, modules)
            quantized = (kind is not None and spec.shape[1] % 128 == 0 and
                         (include_embedding or kind not in modules["convert"].EMBEDDING_KINDS))
            math_selected = donor_spec is not None and kind is not None and (
                not include or any(pattern.search(name) for pattern in include))
            axis = modules["edit_weights"].projection_axis(name) if direction is not None and len(spec.shape) == 2 else None
            if axis and spec.shape[0 if axis == "left" else 1] != width:
                raise ValueError(f"Projection dimension differs from residual width: {name}")
            if axis == "left":
                left_names.append(name)
            elif axis == "right":
                right_names.append(name)
            if math_selected or axis:
                edited_names.append(name)
            cached = cache.load(name, spec.shape) if quantized else None
            print(f"{'Resume' if cached else 'Stream'} {name} {spec.shape} "
                  f"spin8={quantized} math={math_selected} projection={axis}", flush=True)
            if cached is not None:
                packed, payload = cached
            else:
                local_file = work / f"tensor-{index}.bin"
                if quantized:
                    record_array = np.memmap(local_file, mode="w+", dtype=modules["reference_gguf"].GROUP_DTYPE,
                                             shape=(spec.shape[0], spec.shape[1] // 128))
                    dense = None
                else:
                    storage_dtype = np.dtype("<f2") if math_selected or axis else spec.dtype
                    dense = np.memmap(local_file, mode="w+", dtype=storage_dtype, shape=spec.shape)
                progress_started = progress_last = time.monotonic()
                processed_elements = 0
                total_elements = math.prod(spec.shape)
                for selection in _slices(spec.shape, tile_elements, axis, quantized):
                    values = spec.read(selection, modules, max_materialized_bytes, stats)
                    processed_tile_elements = values.size
                    if math_selected or axis:
                        original = values.astype(np.float64)
                        values = original.copy()
                        if math_selected:
                            donor_values = donor_spec.read(selection, modules, max_materialized_bytes, stats).astype(np.float64)
                            values += alpha * (donor_values - original)
                            del donor_values
                        if axis:
                            values = modules["edit_weights"].project_matrix(values, direction, axis)
                        stats["max_edit_tile_bytes"] = max(stats["max_edit_tile_bytes"], values.nbytes)
                        values = values.astype("<f2")
                        del original
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite/overflowing canonical or edited tensor: {name}")
                    if quantized:
                        kwargs = {"scale_steps": scale_steps, "chunk_groups": chunk_groups}
                        if encoder == "torch":
                            kwargs.update(device=device, compute_dtype=getattr(torch, compute_dtype))
                        tile_packed = encode(np.ascontiguousarray(values), bank, **kwargs)
                        rows, columns = selection
                        if axis == "left":
                            columns = slice(columns.start // 128, columns.stop // 128)
                        else:
                            columns = slice(None)
                        destination = record_array[rows, columns]
                        destination["words"] = tile_packed["words"].reshape(destination.shape + (16,))
                        destination["scale"] = tile_packed["scales"].reshape(destination.shape)
                        del tile_packed, destination
                    else:
                        dense[selection] = values
                    del values
                    processed_elements += processed_tile_elements
                    progress_now = time.monotonic()
                    if progress_now - progress_last >= 20 or processed_elements == total_elements:
                        elapsed = max(progress_now - progress_started, 1e-9)
                        print(f"Progress {name}: {processed_elements:,}/{total_elements:,} elements "
                              f"({100 * processed_elements / total_elements:.1f}%), {elapsed:.1f}s elapsed, "
                              f"{processed_elements / elapsed:,.0f} elements/s measured for this tensor", flush=True)
                        progress_last = progress_now
                if quantized:
                    record_array.flush()
                    flat = record_array.reshape(-1)
                    packed = {"codec": codec.CODEC, "words": flat["words"], "scales": flat["scale"],
                              "bank": bank, "shape": spec.shape, "group_size": 128,
                              "hadamard": True, "identity_fallback": True, "uniform_fallback": True}
                    packed, payload = cache.store(name, packed)
                    stats["encoded_tensors_this_run"] += 1
                    del record_array, flat
                    local_file.unlink()
                else:
                    dense.flush()
            if quantized:
                stored_bank = np.array(packed["bank"], dtype="<f8", copy=True)
                descriptors.append((name, payload, gguf.GGMLQuantizationType.SPIN8_G128))
                descriptors.append((name + ".spin8_bank", stored_bank, gguf.GGMLQuantizationType.F64))
                selected_names.append(name)
                records.append({"name": name, "shape": spec.shape, "payload_bytes": payload.nbytes,
                                "uniform_groups": int(np.count_nonzero(packed["scales"] < 0))})
                del packed
            else:
                qtype = gguf.GGMLQuantizationType.F16 if math_selected or axis else spec.qtype
                descriptors.append((name, dense, qtype))
            spec.release()
            if donor_spec is not None:
                donor_spec.release()
            del spec, donor_spec
            gc.collect()
        if not selected_names:
            raise ValueError("No supported 128-aligned matrices were selected")
        if direction is not None:
            if not left_names or not right_names:
                raise ValueError("Direction removal requires both residual writers and token embeddings")
            descriptors.append((modules["edit_weights"].DIRECTION_NAME, direction.astype("<f4"),
                                gguf.GGMLQuantizationType.F32))
            writer.add_uint32("spin8.projection.version", 1)
            writer.add_string("spin8.projection.kind", "residual_direction_removal")
            for key, value in (("left_weights", left_names), ("right_weights", right_names)):
                writer.add_key_value("spin8.projection." + key, value, gguf.GGUFValueType.ARRAY,
                                     sub_type=gguf.GGUFValueType.STRING)
        provenance = {"schema": SCHEMA, "source_revision": source_revision, "donor_revision": donor_revision,
                      "cache_fingerprint": cache.fingerprint, "source_layout": modules["convert"].LAYOUT,
                      "formula": "W_source + alpha * (W_donor - W_source)",
                      "math_alpha": alpha if donor_dir else None, "edited_tensors": edited_names,
                      "operation_order": ["canonical_hf_to_f16", "same_base_math_interpolation",
                                          "residual_direction_projection", "f16_rounding", "spin8_quantization"],
                      "edit_arithmetic_dtype": "float64", "projection_runtime_required": direction is not None,
                      "direction_semantics_verified": False, "answer_data_used": False, "optimizer_steps": 0,
                      "donor_compatibility_checked": "canonical_metadata_tensor_inventory_shape_and_type" if donor_dir else None,
                      "same_base_training_ancestry_verified": False,
                      "quantized_tensors": selected_names, "encoder": encoder, "encoder_device": device,
                      "compute_dtype": compute_dtype if encoder == "torch" else "float64",
                      "encoder_cuda_executed": bool(stats["encoded_tensors_this_run"])
                                               and encoder == "torch" and torch_device.type == "cuda",
                      "cuda_preflight_executed": encoder == "torch" and torch_device.type == "cuda",
                      "quality_measured": False, "full_float_gguf_staged": False, "no_mtp": no_mtp}
        writer.add_uint32("spin8.codec_version", 1)
        writer.add_string("spin8.codec_name", codec.CODEC)
        writer.add_string("spin8.weight_layout", modules["convert"].LAYOUT)
        writer.add_string("spin8.bank_dtype", "float64")
        writer.add_string("spin8.prototype_base_commit", BASE_COMMIT)
        writer.add_string("spin8.export.role", "native_compressed")
        writer.add_string("spin8.export.provenance", json.dumps(provenance, sort_keys=True))
        if edited_names:
            writer.add_string("spin8.edit.role", "direct_weight_edit_no_training")
            writer.add_string("spin8.edit.provenance", json.dumps(provenance, sort_keys=True))
        for name, array, qtype in descriptors:
            writer.add_tensor_info(name, array.shape, array.dtype, array.nbytes, raw_dtype=qtype)
        descriptor, temporary_name = tempfile.mkstemp(prefix="." + output_path.name + ".", dir=output_path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer.write_header_to_file(path=temporary)
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for _, array, _ in descriptors:
                writer.write_tensor_data(array)
            writer.close()
            modules["atomic_output"].publish_file(temporary, output_path, overwrite=force)
        finally:
            writer.close()
            temporary.unlink(missing_ok=True)
    return {"status": "converted", "output": str(output_path), "output_sha256": _sha256(output_path),
            "file_bytes": output_path.stat().st_size, "quantized_tensor_count": len(selected_names),
            "tensors": records, "payload_bpw": 2.125, "runtime_execution_performed": False,
            "math_improvement_measured": False, "refusal_behavior_measured": False, "quality_measured": False,
            "provenance": provenance, "memory": stats,
            "resume": {"directory": str(cache.root), "fingerprint": cache.fingerprint, **cache.stats}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--direction-path", type=Path)
    parser.add_argument("--math-include", action="append")
    parser.add_argument("--source-revision")
    parser.add_argument("--donor-revision")
    parser.add_argument("--encoder", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--bank-kind", choices=("triality", "haar"), default="triality")
    parser.add_argument("--seed", type=int, default=81237)
    parser.add_argument("--scale-steps", type=int, default=17)
    parser.add_argument("--chunk-groups", type=int, default=256)
    parser.add_argument("--tile-elements", type=int, default=1 << 20)
    parser.add_argument("--max-materialized-bytes", type=int, default=2 << 30)
    parser.add_argument("--exclude-embedding", dest="include_embedding", action="store_false")
    parser.add_argument("--include-mtp", dest="no_mtp", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    arguments = vars(parser.parse_args())
    report_path = arguments.pop("report")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = convert_hf_to_spin8(**arguments)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
