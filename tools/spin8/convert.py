#!/usr/bin/env python3
"""Quantize canonical Qwen35 GGUF weights for the private native Spin8 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf
import reference_codec as codec
import reference_gguf
from resume import TensorCache
from atomic_output import publish_file

TYPE = gguf.GGMLQuantizationType.SPIN8_G128
BANK_SUFFIX = ".spin8_bank"
CODEC_VERSION = 1
BASE_COMMIT = "7fce5439fe451cd48340b112ec37d2befe02c776"
LAYOUT = "llama.cpp.qwen35.canonical"
FLOAT_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
DENSE_TYPES = FLOAT_TYPES | {gguf.GGMLQuantizationType.F64, gguf.GGMLQuantizationType.I8,
                            gguf.GGMLQuantizationType.I16, gguf.GGMLQuantizationType.I32,
                            gguf.GGMLQuantizationType.I64}
MATRIX_KINDS = {gguf.MODEL_TENSOR[name] for name in (
    "TOKEN_EMBD", "OUTPUT", "CLS_OUT", "ATTN_Q", "ATTN_K", "ATTN_V", "ATTN_QKV",
    "ATTN_QKV_GATE", "ATTN_GATE", "ATTN_OUT", "FFN_GATE", "FFN_UP", "FFN_DOWN",
    "SSM_BETA", "SSM_ALPHA", "SSM_BETA_ALPHA_CAT", "SSM_OUT", "NEXTN_EH_PROJ",
    "NEXTN_EMBED_TOKENS", "NEXTN_SHARED_HEAD_HEAD")}
EMBEDDING_KINDS = {gguf.MODEL_TENSOR.TOKEN_EMBD, gguf.MODEL_TENSOR.NEXTN_EMBED_TOKENS}
FUSED_WEIGHT_PATTERNS = tuple(
    (kind, re.compile(re.escape(gguf.TENSOR_NAMES[kind]).replace(r"\{bid\}", r"(?:0|[1-9][0-9]*)") + r"\.weight"))
    for kind in (gguf.MODEL_TENSOR.ATTN_QKV_GATE, gguf.MODEL_TENSOR.SSM_BETA_ALPHA_CAT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_value(reader: gguf.GGUFReader, name: str, default: Any = None) -> Any:
    field = reader.get_field(name)
    return default if field is None else field.contents()


def open_source(path: Path, *, native: bool = False) -> gguf.GGUFReader:
    reader = gguf.GGUFReader(path)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError("This prototype requires a little-endian source GGUF")
    if metadata_value(reader, "general.architecture") != "qwen35":
        raise ValueError("Expected an already converted canonical qwen35 GGUF")
    if metadata_value(reader, "split.count", 1) != 1:
        raise ValueError("Merge GGUF shards before this prototype conversion")
    kinds = {tensor.tensor_type for tensor in reader.tensors}
    allowed = DENSE_TYPES | ({TYPE} if native else set())
    if not kinds <= allowed:
        raise ValueError("Input contains unsupported/requantized tensor types: " + str(kinds - allowed))
    if native:
        field = reader.get_field("spin8.codec_version")
        if field is None or field.types != [gguf.GGUFValueType.UINT32] or field.contents() != CODEC_VERSION:
            raise ValueError("Native Spin8 codec_version UINT32=1 is required")
    elif TYPE in kinds or reader.get_field("spin8.codec_version") is not None:
        raise ValueError("Do not requantize an existing native Spin8 model")
    return reader


def logical_shape(tensor: gguf.ReaderTensor) -> tuple[int, ...]:
    return tuple(int(n) for n in reversed(tensor.shape))


def canonical_weight_kind(name: str, names: gguf.TensorNameMap) -> gguf.MODEL_TENSOR | None:
    if not name.endswith(".weight"):
        return None
    mapped = names.get_type_and_name(name, try_suffixes=(".weight",))
    if mapped is not None:
        return mapped[0] if mapped[1] == name else None
    # Converter-created fused tensors have canonical names but no HF alias in TensorNameMap.
    for kind, pattern in FUSED_WEIGHT_PATTERNS:
        if pattern.fullmatch(name):
            return kind
    return None


def projection_kind(tensor: gguf.ReaderTensor, names: gguf.TensorNameMap) -> gguf.MODEL_TENSOR | None:
    shape = logical_shape(tensor)
    kind = canonical_weight_kind(tensor.name, names)
    if (kind not in MATRIX_KINDS or tensor.tensor_type not in FLOAT_TYPES
            or len(shape) != 2 or min(shape) <= 0 or shape[1] % 128):
        return None
    return kind


def resume_settings(args: argparse.Namespace, source_hash: str, selected: list[gguf.ReaderTensor],
                    device: str, torch_module: Any) -> dict[str, Any]:
    code_paths = ["tools/spin8/" + name for name in (
        "convert.py", "resume.py", "atomic_output.py", "reference_codec.py", "reference_gguf.py", "reference_torch.py", "triality_lie.py")]
    code_paths += ["gguf-py/gguf/" + name for name in (
        "constants.py", "gguf_reader.py", "gguf_writer.py", "quants.py", "tensor_mapping.py")]
    dependencies = {"python": sys.version, "numpy": np.__version__}
    if torch_module is not None:
        dependencies.update({"torch": torch_module.__version__, "torch_cuda": torch_module.version.cuda,
                             "torch_git": torch_module.version.git_version,
                             "matmul_precision": torch_module.get_float32_matmul_precision(),
                             "allow_tf32": torch_module.backends.cuda.matmul.allow_tf32})
        if device.startswith("cuda"):
            dependencies["cuda_device_name"] = torch_module.cuda.get_device_name(device)
            dependencies["cuda_device_capability"] = list(torch_module.cuda.get_device_capability(device))
    return {"source_sha256": source_hash, "codec_version": CODEC_VERSION, "base_commit": BASE_COMMIT,
            "encoder": args.encoder, "device": device, "compute_dtype": args.compute_dtype,
            "scale_steps": args.scale_steps, "chunk_groups": args.chunk_groups,
            "seed": args.seed, "bank_kind": args.bank_kind,
            "include": args.include, "exclude": args.exclude, "include_embedding": args.include_embedding,
            "tensors": [{"name": t.name, "shape": logical_shape(t), "source_type": int(t.tensor_type)} for t in selected],
            "dependencies": dependencies, "code_sha256": {name: sha256(ROOT / name) for name in code_paths}}


def dense_input(tensor: gguf.ReaderTensor, work: Path) -> np.ndarray:
    if tensor.tensor_type != gguf.GGMLQuantizationType.BF16:
        return tensor.data
    shape = logical_shape(tensor)
    raw = tensor.data.reshape(-1).view("<u2")
    target = np.memmap(work / "bf16-input.f32", mode="w+", dtype="<f4", shape=shape)
    flat = target.reshape(-1)
    for start in range(0, len(raw), 1 << 20):
        stop = min(start + (1 << 20), len(raw))
        bits = raw[start:stop].astype("<u4") << 16
        flat[start:stop] = bits.view("<f4")
    target.flush()
    return target


def store_packed(packed: dict[str, Any], path: Path) -> np.memmap:
    codec._validate_packed(packed)
    rows, columns = packed["shape"]
    groups = rows * columns // 128
    output = np.memmap(path, mode="w+", dtype=reference_gguf.GROUP_DTYPE, shape=(groups,))
    for start in range(0, groups, 65536):
        stop = min(start + 65536, groups)
        output["words"][start:stop] = packed["words"][start:stop]
        output["scale"][start:stop] = packed["scales"][start:stop]
    output.flush()
    return np.memmap(path, mode="r", dtype="u1", shape=(rows, columns // 128 * 34))


def packed_view(payload: np.ndarray, bank: np.ndarray, shape: tuple[int, ...]) -> dict[str, Any]:
    if len(shape) != 2 or shape[1] % 128:
        raise ValueError("Native Spin8 weights must be matrices with 128-aligned rows")
    if bank.shape != (9, 8, 8) or bank.dtype != np.dtype("<f8"):
        raise ValueError("Native bank must be an F64 [9,8,8] tensor")
    records = payload.reshape(-1).view(reference_gguf.GROUP_DTYPE)
    result = {"codec": codec.CODEC, "words": records["words"], "scales": records["scale"],
              "bank": bank, "shape": shape, "group_size": 128, "hadamard": True,
              "identity_fallback": True, "uniform_fallback": True}
    codec._validate_packed(result)
    return result


def decode_to_file(packed: dict[str, Any], path: Path) -> np.memmap:
    words, scales, bank, shape = codec._validate_packed(packed)
    output = np.memmap(path, mode="w+", dtype="<f4", shape=shape)
    groups = output.reshape(-1, 128)
    for start in range(0, len(words), 256):
        stop = min(start + 256, len(words))
        groups[start:stop] = codec._decode_groups(words[start:stop], scales[start:stop], bank)
    output.flush()
    return np.memmap(path, mode="r", dtype="<f4", shape=shape)


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, *, native: bool) -> None:
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.file_type", "general.alignment"):
            continue
        if name.startswith("spin8.") and not name.startswith(("spin8.fixture.", "spin8.projection.", "spin8.edit.")):
            continue
        if name.startswith("split."):
            continue
        subtype = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), field.types[0], sub_type=subtype)
    if native:
        writer.add_uint32("spin8.codec_version", CODEC_VERSION)
        writer.add_string("spin8.codec_name", codec.CODEC)
        writer.add_string("spin8.weight_layout", LAYOUT)
        writer.add_string("spin8.bank_dtype", "float64")
        writer.add_string("spin8.prototype_base_commit", BASE_COMMIT)


def write_model(path: Path, source: gguf.GGUFReader, descriptors: list[tuple[str, np.ndarray, gguf.GGMLQuantizationType]],
                provenance: dict[str, Any], *, native: bool, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, "qwen35")
        copy_metadata(source, writer, native=native)
        writer.add_string("spin8.export.provenance", json.dumps(provenance, sort_keys=True, separators=(",", ":")))
        writer.add_string("spin8.export.role", "native_compressed" if native else "decoded_control")
        for name, array, kind in descriptors:
            writer.add_tensor_info(name, array.shape, array.dtype, array.nbytes, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for _, array, _ in descriptors:
            writer.write_tensor_data(array)
        writer.close()
        writer = None
        publish_file(temporary, path, overwrite=overwrite)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def check_paths(source: Path, targets: list[Path | None], overwrite: bool) -> None:
    actual = [path.resolve() for path in targets if path is not None]
    if source.resolve() in actual or len(set(actual)) != len(actual):
        raise ValueError("Source and output files must be distinct")
    for path in actual:
        if path.exists() and not overwrite:
            raise FileExistsError(path)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    check_paths(args.source, [args.output, args.decoded_control, getattr(args, "reference_output", None)], args.force)
    if args.command == "transcode":
        targets = [path.resolve() for path in (args.output, args.decoded_control) if path is not None]
        if args.reference.resolve() in targets:
            raise ValueError("Reference input and output files must be distinct")
    encoder = getattr(args, "encoder", "numpy")
    device = getattr(args, "device", "cpu")
    compute_dtype = getattr(args, "compute_dtype", "float32")
    torch_encoder = None
    torch_module = None
    if args.command == "quantize" and encoder == "torch":
        try:
            import torch
            from reference_torch import encode_torch
        except ImportError as error:
            raise ValueError("Install PyTorch for --encoder torch") from error
        device = str(torch.device(device))
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA encoding requested, but this PyTorch runtime has no available CUDA device")
        torch_encoder = encode_torch
        torch_module = torch
    elif args.command == "quantize" and device != "cpu":
        raise ValueError("Use --encoder torch for a CUDA device")
    reader = open_source(args.source)
    source_hash = sha256(args.source)
    original_names = {tensor.name for tensor in reader.tensors}
    if any(name.endswith(BANK_SUFFIX) for name in original_names):
        raise ValueError("Source already contains Spin8 auxiliary banks")
    overlay = None
    if args.command == "transcode":
        overlay, metadata = reference_gguf.read_gguf(args.reference)
        if metadata.get("project.weight_layout") != LAYOUT or metadata.get("project.canonical_gguf_sha256") != source_hash:
            raise ValueError("Reference lacks a matching canonical-layout/source hash. HF-layout or partial-row pilots cannot be renamed into native GGUF.")
        if not set(overlay) <= original_names or not all(isinstance(value, dict) for value in overlay.values()):
            raise ValueError("Reference overlay must contain only exact canonical packed tensor names")
    include = [re.compile(pattern) for pattern in getattr(args, "include", [])]
    exclude = [re.compile(pattern) for pattern in getattr(args, "exclude", [])]
    blocks = metadata_value(reader, "qwen35.block_count")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= 1024:
        raise ValueError("Expected qwen35.block_count in [1,1024]")
    names = gguf.TensorNameMap(gguf.MODEL_ARCH.QWEN35, blocks)
    selection = []
    for tensor in reader.tensors:
        kind = projection_kind(tensor, names)
        if overlay is not None:
            if tensor.name in overlay:
                if kind is None or tuple(overlay[tensor.name]["shape"]) != logical_shape(tensor):
                    raise ValueError("Reference must match a complete supported canonical projection: " + tensor.name)
                selection.append(tensor)
        elif (kind is not None and (not include or any(p.search(tensor.name) for p in include))
              and not any(p.search(tensor.name) for p in exclude)
              and (args.include_embedding or kind not in EMBEDDING_KINDS)):
            selection.append(tensor)
    if not selection:
        raise ValueError("No complete 128-aligned supported projection matched the selection")
    selection_names = {tensor.name for tensor in selection}
    cache = None
    if getattr(args, "resume_dir", None) is not None:
        cache = TensorCache(args.resume_dir, resume_settings(args, source_hash, selection, device, torch_module))
    bank = codec.make_bank(getattr(args, "bank_kind", "triality"), getattr(args, "seed", 81237))
    selected_names, tensor_info = [], []
    provenance = {"source_name": args.source.name, "source_sha256": source_hash,
                  "source_architecture": "qwen35", "source_layout": LAYOUT, "codec": codec.CODEC,
                  "private_ggml_type": 44, "command": args.command,
                  "encoder": encoder if args.command == "quantize" else "canonical_reference_transcode",
                  "encoder_device": device if args.command == "quantize" else None,
                  "compute_dtype": compute_dtype if encoder == "torch" else "float64",
                  "encoder_cuda_executed": args.command == "quantize" and encoder == "torch" and device.startswith("cuda"),
                  "scale_steps": getattr(args, "scale_steps", None), "seed": getattr(args, "seed", None),
                  "bank_kind": getattr(args, "bank_kind", None)}
    with tempfile.TemporaryDirectory(prefix="spin8-native-", dir=args.temp_dir) as directory:
        work = Path(directory)
        native_descriptors, control_descriptors, references = [], [], {}
        for index, tensor in enumerate(reader.tensors):
            shape = logical_shape(tensor)
            if tensor.name not in selection_names:
                descriptor = (tensor.name, tensor.data, tensor.tensor_type)
                native_descriptors.append(descriptor)
                control_descriptors.append(descriptor)
                continue
            print("Spin8 " + args.command + " " + tensor.name + " " + str(shape), flush=True)
            cached = cache.load(tensor.name, shape) if cache is not None else None
            if cached is not None:
                packed, payload = cached
                print("Verified resume cache hit: " + tensor.name, flush=True)
            elif overlay is None:
                values = dense_input(tensor, work)
                if torch_encoder is None:
                    packed = codec.encode(values, bank, scale_steps=args.scale_steps, chunk_groups=args.chunk_groups)
                else:
                    packed = torch_encoder(values, bank, scale_steps=args.scale_steps, chunk_groups=args.chunk_groups,
                                           device=device, compute_dtype=getattr(torch, compute_dtype))
                del values
            else:
                packed = overlay[tensor.name]
            if cache is not None and cached is None:
                packed, payload = cache.store(tensor.name, packed)
            elif cache is None:
                payload = store_packed(packed, work / f"tensor-{index}.spin8")
            stored_bank = np.array(packed["bank"], dtype="<f8", copy=True)
            view = packed_view(payload, stored_bank, shape)
            references[tensor.name] = view
            native_descriptors.append((tensor.name, payload, TYPE))
            native_descriptors.append((tensor.name + BANK_SUFFIX, stored_bank, gguf.GGMLQuantizationType.F64))
            if args.decoded_control is not None:
                dense = decode_to_file(view, work / f"tensor-{index}.decoded-f32")
                control_descriptors.append((tensor.name, dense, gguf.GGMLQuantizationType.F32))
            selected_names.append(tensor.name)
            tensor_info.append({"name": tensor.name, "shape": shape, "payload_bytes": payload.nbytes,
                                "bank_bytes": stored_bank.nbytes, "uniform_groups": int(np.count_nonzero(view["scales"] < 0))})
            del packed
        if not selected_names:
            raise ValueError("No complete 128-aligned matrix matched the selection")
        provenance["quantized_tensors"] = selected_names
        if overlay is not None:
            provenance["reference_sha256"] = sha256(args.reference)
        write_model(args.output, reader, native_descriptors, provenance, native=True, overwrite=args.force)
        if args.decoded_control is not None:
            write_model(args.decoded_control, reader, control_descriptors, provenance, native=False, overwrite=args.force)
        if getattr(args, "reference_output", None) is not None:
            reference_gguf.write_gguf(args.reference_output, references,
                {"project.weight_layout": LAYOUT, "project.canonical_gguf_sha256": source_hash,
                 "project.canonical_source": args.source.name, "project.artifact_kind": "CANONICAL_WEIGHT_OVERLAY"},
                overwrite=args.force)
    report = {"status": "converted", "output": str(args.output), "output_sha256": sha256(args.output),
              "file_bytes": args.output.stat().st_size, "source_sha256": source_hash,
              "quantized_tensor_count": len(selected_names), "tensors": tensor_info,
              "payload_bpw": 2.125, "runtime_execution_performed": False,
              "quality_measured": False, "provenance": provenance}
    if cache is not None:
        report["resume"] = {"directory": str(cache.root), "fingerprint": cache.fingerprint, **cache.stats}
    if args.decoded_control is not None:
        report["decoded_control"] = {"path": str(args.decoded_control), "sha256": sha256(args.decoded_control), "file_bytes": args.decoded_control.stat().st_size}
    return report


def decode_model(args: argparse.Namespace) -> dict[str, Any]:
    check_paths(args.source, [args.output], args.force)
    reader = open_source(args.source, native=True)
    lookup = {tensor.name: tensor for tensor in reader.tensors}
    bank_names = {tensor.name + BANK_SUFFIX for tensor in reader.tensors if tensor.tensor_type == TYPE}
    if not bank_names or not bank_names <= lookup.keys():
        raise ValueError("Native compressed tensor or bank is missing")
    if {name for name in lookup if name.endswith(BANK_SUFFIX)} != bank_names:
        raise ValueError("Unexpected auxiliary Spin8 bank")
    source_hash = sha256(args.source)
    with tempfile.TemporaryDirectory(prefix="spin8-decode-", dir=args.temp_dir) as directory:
        descriptors = []
        for index, tensor in enumerate(reader.tensors):
            if tensor.name in bank_names:
                continue
            if tensor.tensor_type == TYPE:
                bank = lookup[tensor.name + BANK_SUFFIX]
                if bank.tensor_type != gguf.GGMLQuantizationType.F64:
                    raise ValueError("Bank tensor must use GGML F64")
                packed = packed_view(tensor.data, bank.data, logical_shape(tensor))
                array = decode_to_file(packed, Path(directory) / f"tensor-{index}.f32")
                descriptors.append((tensor.name, array, gguf.GGMLQuantizationType.F32))
            else:
                descriptors.append((tensor.name, tensor.data, tensor.tensor_type))
        provenance = {"command": "decode", "native_source_sha256": source_hash, "codec": codec.CODEC}
        write_model(args.output, reader, descriptors, provenance, native=False, overwrite=args.force)
    return {"status": "decoded", "source_sha256": source_hash, "output": str(args.output), "output_sha256": sha256(args.output),
            "runtime_execution_performed": False, "quality_measured": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quantize = sub.add_parser("quantize", help="Quantize complete canonical GGUF matrices")
    transcode = sub.add_parser("transcode", help="Overlay a source-bound canonical reference container")
    decode = sub.add_parser("decode", help="Create an F32 decoded control from a native model")
    for item in (quantize, transcode, decode):
        item.add_argument("source", type=Path)
        if item is transcode:
            item.add_argument("reference", type=Path)
        item.add_argument("output", type=Path)
        item.add_argument("--force", action="store_true")
        item.add_argument("--temp-dir", type=Path)
    for item in (quantize, transcode):
        item.add_argument("--decoded-control", type=Path)
    quantize.add_argument("--reference-output", type=Path)
    quantize.add_argument("--resume-dir", type=Path, help="Verified per-tensor checkpoints; one immutable configuration and one writer per directory")
    quantize.add_argument("--include", action="append", default=[], help="Tensor name regex; repeatable")
    quantize.add_argument("--exclude", action="append", default=[], help="Tensor name regex; repeatable")
    quantize.add_argument("--include-embedding", action="store_true")
    quantize.add_argument("--bank-kind", choices=["triality", "haar"], default="triality")
    quantize.add_argument("--seed", type=int, default=81237)
    quantize.add_argument("--scale-steps", type=int, default=17)
    quantize.add_argument("--chunk-groups", type=int, default=256)
    quantize.add_argument("--encoder", choices=["numpy", "torch"], default="numpy")
    quantize.add_argument("--device", default="cpu", help="Encoder device, e.g. cpu or cuda; CUDA requires --encoder torch")
    quantize.add_argument("--compute-dtype", choices=["float32", "float64"], default="float32", help="PyTorch search precision; final scales remain FP16")
    args = parser.parse_args()
    try:
        result = decode_model(args) if args.command == "decode" else convert(args)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    report_path = args.output.with_suffix(args.output.suffix + ".conversion.json")
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
