#!/usr/bin/env python3
"""Direct floating-weight arithmetic and residual-direction removal, with no training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from atomic_output import publish_file
from convert import (gguf, FLOAT_TYPES, MATRIX_KINDS, canonical_weight_kind, copy_metadata, logical_shape,
                     metadata_value, open_source, sha256)

PREFIX = "spin8.projection."
DIRECTION_NAME = PREFIX + "direction"
LEFT_PATTERN = re.compile(r"^blk\.[0-9]+\.(?:attn_output|ffn_down|ssm_out)\.weight$")
RIGHT_NAMES = {"token_embd.weight"}


def float_view(tensor: gguf.ReaderTensor, selection) -> np.ndarray:
    shape = logical_shape(tensor)
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        bits = tensor.data.reshape(-1).view("<u2").reshape(shape)[selection].astype("<u4") << 16
        return bits.view("<f4").astype(np.float64)
    return np.array(tensor.data.reshape(shape)[selection], dtype=np.float64, copy=True)


def normalize_direction(path: Path, width: int) -> np.ndarray:
    raw = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(raw, np.ndarray) or raw.shape != (width,) or raw.dtype.kind not in "fi" or not np.isfinite(raw).all():
        raise ValueError("Direction must be one finite numeric residual vector with the model's hidden width")
    value = raw.astype(np.float64)
    scale = np.max(np.abs(value))
    if scale == 0:
        raise ValueError("Zero direction cannot define a projection")
    value /= scale
    value /= np.linalg.norm(value)
    # The serialized F32 vector is the contract used by both editing and inference.
    value = value.astype(np.float32).astype(np.float64)
    return value


def projection_axis(name: str) -> str | None:
    if LEFT_PATTERN.fullmatch(name):
        return "left"
    if name in RIGHT_NAMES:
        return "right"
    return None


def project_matrix(matrix: np.ndarray, direction: np.ndarray, axis: str) -> np.ndarray:
    norm2 = float(direction @ direction)
    if not math.isfinite(norm2) or norm2 <= 0:
        raise ValueError("Invalid direction norm")
    if axis == "left":
        return matrix - direction[:, None] * ((direction @ matrix) / norm2)[None, :]
    if axis == "right":
        return matrix - ((matrix @ direction) / norm2)[:, None] * direction[None, :]
    raise ValueError("Projection axis must be left or right")


def validate_unfolded(reader: gguf.GGUFReader) -> None:
    if any(key.startswith("prism.hadamard.") for key in reader.fields):
        raise ValueError("Weight editing requires an unfolded canonical GGUF; latent Hadamard bases are not interchangeable")
    if any(key.startswith(PREFIX) for key in reader.fields) or any(t.name == DIRECTION_NAME for t in reader.tensors):
        raise ValueError("Input already contains a projection; use the original float model and one explicit edit recipe")


def validate_metadata(reference: gguf.GGUFReader, candidate: gguf.GGUFReader, label: str) -> None:
    keys = {key for key in reference.fields if key.startswith(("tokenizer.", "qwen35."))}
    keys |= {key for key in candidate.fields if key.startswith(("tokenizer.", "qwen35."))}
    differences = []
    for key in sorted(keys):
        left, right = reference.get_field(key), candidate.get_field(key)
        if left is None or right is None or left.types != right.types or left.contents() != right.contents():
            differences.append(key)
    if differences:
        raise ValueError(label + " metadata/basis differs: " + ", ".join(differences[:12]) +
                         ". Resolve tokenizer/export semantics before subtracting weights.")


def edit(args: argparse.Namespace) -> dict[str, Any]:
    if not args.math_donor and not args.remove_direction:
        raise ValueError("Specify --math-donor/--math-base, --remove-direction, or both")
    if bool(args.math_donor) != bool(args.math_base):
        raise ValueError("A task vector requires both --math-base and --math-donor")
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        raise ValueError("This initial interpolation experiment accepts 0 <= alpha <= 1")
    if args.tile_elements < 128:
        raise ValueError("tile-elements must be at least 128")
    sources = [args.source, args.math_base, args.math_donor, args.remove_direction]
    if args.output.resolve() in {path.resolve() for path in sources if path}:
        raise ValueError("Output must not overwrite an input")
    if args.output.exists() and not args.force:
        raise FileExistsError(args.output)
    source = open_source(args.source)
    validate_unfolded(source)
    width = metadata_value(source, "qwen35.embedding_length")
    layers = metadata_value(source, "qwen35.block_count")
    if type(width) is not int or not 1 <= width <= 65536 or type(layers) is not int or not 1 <= layers <= 1024:
        raise ValueError("Expected bounded integer embedding_length and block_count metadata")
    names = gguf.TensorNameMap(gguf.MODEL_ARCH.QWEN35, layers)
    source_hash = sha256(args.source)
    math_base = math_donor = None
    base_lookup = donor_lookup = {}
    if args.math_donor:
        math_base = open_source(args.math_base)
        math_donor = open_source(args.math_donor)
        validate_unfolded(math_base)
        validate_unfolded(math_donor)
        validate_metadata(source, math_base, "Math base")
        validate_metadata(source, math_donor, "Math donor")
        base_lookup = {tensor.name: tensor for tensor in math_base.tensors}
        donor_lookup = {tensor.name: tensor for tensor in math_donor.tensors}
    include = [re.compile(pattern) for pattern in args.math_include]
    preserve_missing = [re.compile(pattern) for pattern in args.preserve_missing]
    direction = normalize_direction(args.remove_direction, width) if args.remove_direction else None
    dtype = np.dtype("<f4" if args.outtype == "f32" else "<f2")
    output_type = gguf.GGMLQuantizationType.F32 if args.outtype == "f32" else gguf.GGMLQuantizationType.F16
    recipe = {"schema": "spin8-direct-weight-edit-v1", "source_sha256": source_hash,
              "operation_order": ["task_vector_on_selected_linear_weights", "residual_direction_projection"],
              "answer_data_used": False, "optimizer_steps": 0, "quality_measured": False,
              "math_alpha": args.alpha if math_donor else None,
              "math_include": args.math_include, "preserve_missing": args.preserve_missing,
              "edited_storage_dtype": args.outtype, "edit_arithmetic_dtype": "float64",
              "projection_runtime_required": direction is not None,
              "projection_role": "refusal_candidate" if direction is not None else None,
              "direction_semantics_verified": False}
    if math_donor:
        recipe["math_base_sha256"] = sha256(args.math_base)
        recipe["math_donor_sha256"] = sha256(args.math_donor)
    if direction is not None:
        recipe["direction_file_sha256"] = sha256(args.remove_direction)
        recipe["direction_f32_sha256"] = hashlib.sha256(direction.astype("<f4").tobytes()).hexdigest()
    edited_records, left_names, right_names, skipped_missing = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="spin8-weight-edit-", dir=args.temp_dir) as directory:
        work = Path(directory)
        descriptors = []
        for index, tensor in enumerate(source.tensors):
            name, shape = tensor.name, logical_shape(tensor)
            matrix = tensor.tensor_type in FLOAT_TYPES and len(shape) == 2
            math_kind = canonical_weight_kind(name, names)
            math_selected = bool(math_donor and matrix and math_kind in MATRIX_KINDS and
                                 (not include or any(pattern.search(name) for pattern in include)))
            if math_selected and (name not in base_lookup or name not in donor_lookup):
                if not any(pattern.fullmatch(name) for pattern in preserve_missing):
                    raise ValueError("Missing task-vector tensor: " + name)
                math_selected = False
                skipped_missing.append(name)
            if math_selected:
                for reference in (base_lookup[name], donor_lookup[name]):
                    if reference.tensor_type not in FLOAT_TYPES or logical_shape(reference) != shape:
                        raise ValueError("Task-vector tensor type/shape differs: " + name)
            axis = projection_axis(name) if direction is not None and matrix else None
            if axis and (shape[0 if axis == "left" else 1] != width):
                raise ValueError("Projection dimension differs from residual width: " + name)
            if not math_selected and not axis:
                descriptors.append((name, tensor.data, tensor.tensor_type))
                continue
            print("Editing", name, "task_delta=" + str(math_selected), "projection=" + str(axis), flush=True)
            output = np.memmap(work / f"{index:05d}.weights", mode="w+", dtype=dtype, shape=shape)
            tile_axis = 1 if axis == "left" else 0
            other = shape[1 - tile_axis]
            step = max(1, args.tile_elements // other)
            source_squared = delta_squared = edited_squared = 0.0
            leakage_squared = 0.0
            for start in range(0, shape[tile_axis], step):
                slices = [slice(None), slice(None)]
                slices[tile_axis] = slice(start, min(start + step, shape[tile_axis]))
                slices = tuple(slices)
                original = float_view(tensor, slices)
                changed = original.copy()
                if math_selected:
                    changed += args.alpha * (float_view(donor_lookup[name], slices) - float_view(base_lookup[name], slices))
                if axis:
                    changed = project_matrix(changed, direction, axis)
                stored = changed.astype(dtype)
                if not np.isfinite(stored).all():
                    raise ValueError("Non-finite or overflowing edited tensor: " + name)
                output[slices] = stored
                stored64 = stored.astype(np.float64)
                source_squared += float(np.sum(original * original))
                delta_squared += float(np.sum((stored64 - original) ** 2))
                edited_squared += float(np.sum(stored64 * stored64))
                if axis:
                    component = direction @ stored64 if axis == "left" else stored64 @ direction
                    leakage_squared += float(component @ component) / float(direction @ direction)
            output.flush()
            descriptors.append((name, output, output_type))
            edited_records.append({"name": name, "shape": shape, "task_delta_applied": math_selected,
                                   "projection_axis": axis,
                                   "relative_weight_change": math.sqrt(delta_squared / source_squared) if source_squared else None,
                                   "stored_projection_energy_fraction": leakage_squared / edited_squared if axis and edited_squared else 0.0})
            if axis == "left":
                left_names.append(name)
            if axis == "right":
                right_names.append(name)
        if direction is not None and (not left_names or not right_names):
            raise ValueError("Refusal-direction removal requires residual writers and the token embedding table")
        if not edited_records:
            raise ValueError("No editable weight matrix was selected")
        if direction is not None:
            descriptors.append((DIRECTION_NAME, direction.astype("<f4"), gguf.GGMLQuantizationType.F32))
        recipe["edited_tensors"] = edited_records
        recipe["preserved_missing_tensors"] = skipped_missing
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="." + args.output.name + ".", dir=args.output.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        writer = None
        try:
            writer = gguf.GGUFWriter(temporary, "qwen35")
            copy_metadata(source, writer, native=False)
            writer.add_string("spin8.edit.provenance", json.dumps(recipe, sort_keys=True, separators=(",", ":")))
            writer.add_string("spin8.edit.role", "direct_weight_edit_no_training")
            if direction is not None:
                writer.add_uint32(PREFIX + "version", 1)
                writer.add_string(PREFIX + "kind", "residual_direction_removal")
                writer.add_key_value(PREFIX + "left_weights", left_names, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
                writer.add_key_value(PREFIX + "right_weights", right_names, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
            for name, array, kind in descriptors:
                writer.add_tensor_info(name, array.shape, array.dtype, array.nbytes, raw_dtype=kind)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for _, array, _ in descriptors:
                writer.write_tensor_data(array)
            writer.close()
            writer = None
            publish_file(temporary, args.output, overwrite=args.force)
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
    return {"status": "WEIGHTS_EDITED", "output": str(args.output), "output_sha256": sha256(args.output),
            "math_improvement_measured": False, "refusal_behavior_measured": False, "recipe": recipe}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--math-base", type=Path)
    parser.add_argument("--math-donor", type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--math-include", action="append", default=[])
    parser.add_argument("--preserve-missing", action="append", default=[], help="Full-match regex of target tensors explicitly preserved if absent from a donor")
    parser.add_argument("--remove-direction", type=Path, help="Model-specific residual direction .npy, not a fabricated default")
    parser.add_argument("--outtype", choices=("f16", "f32"), default="f16")
    parser.add_argument("--tile-elements", type=int, default=1 << 20)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = edit(args)
    report_path = Path(str(args.output) + ".edit.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "report": str(report_path)}))


if __name__ == "__main__":
    main()
