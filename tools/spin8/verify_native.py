#!/usr/bin/env python3
"""Compare native Spin8 logits against the decoded F32 version of the same weights."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_logits(metadata: dict) -> array.array:
    path = Path(metadata["logits_file"])
    shape = metadata["logits_shape"]
    count = math.prod(shape)
    if len(shape) != 2 or count <= 0 or path.stat().st_size != count * 4:
        raise ValueError("invalid logit shape or byte length")
    values = array.array("f")
    if values.itemsize != 4:
        raise RuntimeError("this Python does not use 32-bit float arrays")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    dtype = metadata["logits_dtype"]
    if dtype not in ("<f4", ">f4"):
        raise ValueError("unsupported logit dtype")
    if (dtype == "<f4") != (sys.byteorder == "little"):
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("logits contain non-finite values")
    return values


def invoke(args: argparse.Namespace, model: Path, name: str, ngl: int, device: str) -> dict:
    prefix = args.out_dir / name
    command = [
        str(args.binary), "--model", str(model), "--output-prefix", str(prefix),
        "--ngl", str(ngl), "--threads", str(args.threads), "--ctx", str(args.ctx),
        "--cache-type", args.cache_type,
        "--flash-attn", args.flash_attn, "--prompt-tokens", args.prompt_tokens,
        "--decode-tokens", args.decode_tokens,
    ]
    if device:
        command += ["--device", device]
    if args.mmap is not None:
        command += ["--mmap" if args.mmap else "--no-mmap"]
    result = subprocess.run(command, check=False)
    metadata_path = Path(str(prefix) + ".json")
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited {result.returncode}; inspect {prefix}.log and .json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "PASS" or metadata.get("finite_logits") is not True:
        raise RuntimeError(f"{name} did not report finite successful execution")
    metadata["command"] = command
    metadata["model_artifact"] = artifact(model)
    metadata["logits_artifact"] = artifact(Path(metadata["logits_file"]))
    metadata["metadata_artifact"] = artifact(metadata_path)
    metadata["runtime_library_artifacts"] = [
        artifact(Path(path)) for path in metadata.get("loaded_runtime_libraries", [])
    ]
    return metadata


def source_identity() -> dict:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"])
        status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
        return {
            "path": str(repo), "git_commit": commit, "working_tree_dirty": bool(status),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "note": "Working-tree identity is contextual; runtime executable and loaded libraries are hashed separately.",
        }
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(repo), "git_commit": None}


def compare(args: argparse.Namespace, actual: dict, control: dict) -> dict:
    for key in ("prompt_tokens", "decode_tokens", "logits_shape", "n_vocab", "n_embd", "n_layer",
                "cache_type_k", "cache_type_v"):
        if actual[key] != control[key]:
            raise ValueError(f"execution mismatch: {key}")
    for metadata in (actual, control):
        if metadata["cache_type_k"] != args.cache_type or metadata["cache_type_v"] != args.cache_type:
            raise ValueError("execution cache types differ from --cache-type")
    row_ids = lambda value: [(row["phase"], row["position"], row["input_token"]) for row in value["rows"]]
    if row_ids(actual) != row_ids(control):
        raise ValueError("logit row identities do not match")
    values = read_logits(actual)
    reference = read_logits(control)
    maximum = 0.0
    squared_error = 0.0
    reference_squared = 0.0
    actual_squared = 0.0
    dot = 0.0
    violations = 0
    for value, expected in zip(values, reference):
        error = abs(value - expected)
        maximum = max(maximum, error)
        squared_error += error * error
        reference_squared += expected * expected
        actual_squared += value * value
        dot += value * expected
        violations += error > args.atol + args.rtol * abs(expected)
    cosine = dot / math.sqrt(reference_squared * actual_squared) if reference_squared and actual_squared else None
    placements = [
        item for item in actual["observed_output_buffer_placements"]
        if "SPIN8" in item["op"] and item["completed_observations"] > 0
    ]
    if not placements:
        raise ValueError("no completed Spin8 graph operation was observed")
    checked_placements = [
        item for item in placements if args.require_spin8_op == "all" or item["op"] == args.require_spin8_op
    ]
    if not checked_placements:
        raise ValueError(f"no completed {args.require_spin8_op} graph operation was observed")
    device_verified = None
    if args.require_spin8_device:
        expected = args.require_spin8_device.casefold()
        device_verified = all(expected in item["device"].casefold() for item in checked_placements)
    return {
        "status": "PASS" if violations == 0 and device_verified is not False else "FAIL",
        "scope": "native graph parity against decoded codec weights; not source-BF16 accuracy or language quality",
        "atol": args.atol, "rtol": args.rtol,
        "cache_type_k": actual["cache_type_k"], "cache_type_v": actual["cache_type_v"],
        "elements": len(values), "tolerance_violations": violations,
        "max_absolute_error": maximum,
        "rmse": math.sqrt(squared_error / len(values)),
        "relative_l2_error": math.sqrt(squared_error / reference_squared) if reference_squared else None,
        "cosine_similarity": cosine,
        "top1_equal_rows": sum(a["top1_token"] == b["top1_token"] for a, b in zip(actual["rows"], control["rows"])),
        "rows": len(actual["rows"]),
        "spin8_output_placements": placements,
        "required_spin8_device": args.require_spin8_device,
        "required_spin8_op": args.require_spin8_op,
        "checked_device_placements": checked_placements,
        "all_observed_required_op_output_devices_match": device_verified,
        "placement_note": actual["placement_note"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="native Spin8 GGUF")
    parser.add_argument("--control", required=True, type=Path, help="F32 GGUF decoded from the same Spin8 weights")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--control-ngl", type=int, default=None, help="control GPU layers; defaults to --ngl; use 0 for CPU F32 reference")
    parser.add_argument("--device", default="")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--cache-type", choices=("f16", "f32"), default="f16",
                        help="K/V cache storage type for both executions; use f32 to isolate cache rounding")
    parser.add_argument("--flash-attn", choices=("off", "on", "auto"), default="off")
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--mmap", dest="mmap", action="store_true", default=None)
    loading.add_argument("--no-mmap", dest="mmap", action="store_false")
    parser.add_argument("--prompt-tokens", default="1,4,7,9,13")
    parser.add_argument("--decode-tokens", default="17,21,25")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--require-spin8-device", default="", help="case-insensitive substring in each checked Spin8 op output device")
    parser.add_argument("--require-spin8-op", default="MUL_MAT_SPIN8", choices=("MUL_MAT_SPIN8", "GET_ROWS_SPIN8", "all"))
    args = parser.parse_args()
    for key in ("binary", "model", "control"):
        path = getattr(args, key).resolve(strict=True)
        if not path.is_file():
            parser.error(f"{key} is not a file")
        setattr(args, key, path)
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0:
        parser.error("tolerances must be finite and non-negative")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "spin8-native-parity-v1", "status": "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_executable": artifact(args.binary), "source": source_identity(),
        "precision_environment": {
            name: os.environ.get(name)
            for name in ("GGML_CUDA_SPIN8_VALIDATE_F32", "NVIDIA_TF32_OVERRIDE",
                         "GGML_CUDA_CUBLAS_COMPUTE_TYPE")
        },
    }
    try:
        control_ngl = args.ngl if args.control_ngl is None else args.control_ngl
        control_device = "" if control_ngl == 0 else args.device
        report["spin8"] = invoke(args, args.model, "spin8", args.ngl, args.device)
        report["control"] = invoke(args, args.control, "decoded_control", control_ngl, control_device)
        report["comparison"] = compare(args, report["spin8"], report["control"])
        report["status"] = report["comparison"]["status"]
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report["error"] = str(error)
    output = args.out_dir / "parity.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output), "comparison": report.get("comparison"), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
