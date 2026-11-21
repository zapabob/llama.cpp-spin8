#!/usr/bin/env python3
"""Verify optional native weight projection against independently folded F32 weights.

Uses a synthetic direction and a tiny fixture. This tests linear algebra and
runtime integration, not refusal behavior or model capability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

import convert as bridge
import gguf
from test_loader import invoke as invoke_loader
from verify_native import artifact

HERE = Path(__file__).resolve().parent
PREFIX = "spin8.projection."
DIRECTION = PREFIX + "direction"
NEGATIVE_CASES = (
    "missing_version", "wrong_version_type", "unsupported_version",
    "missing_direction", "wrong_direction_type", "wrong_direction_width",
    "nonfinite_direction", "zero_direction", "invalid_left_name",
    "invalid_right_name", "missing_target", "duplicate_target",
    "wrong_array_type", "empty_targets",
)


def run(command: list[str], log: Path, timeout: int = 240) -> None:
    with log.open("wb") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                 check=False, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def write_variant(reader: gguf.GGUFReader, path: Path, direction: np.ndarray,
                  left: list[str], right: list[str], *, folded: bool = False,
                  tied: bool = False, malformed: str = "") -> dict:
    left, right = list(left), list(right)
    if malformed == "invalid_left_name":
        left = ["output.weight"]
    elif malformed == "invalid_right_name":
        right = ["output.weight"]
    elif malformed == "missing_target":
        left = ["blk.99.ffn_down.weight"]
    elif malformed == "duplicate_target":
        left += left[:1]
    elif malformed == "empty_targets":
        left, right = [], []
    stored = np.asarray(direction, dtype="<f4").copy()
    if malformed == "wrong_direction_width":
        stored = stored[:-1]
    elif malformed == "nonfinite_direction":
        stored[0] = np.nan
    elif malformed == "zero_direction":
        stored[:] = 0
    # Match load-time normalization, then independently materialize I-u*u^T.
    unit = stored.astype(np.float64)
    if folded:
        unit = (unit / np.linalg.norm(unit)).astype(np.float32).astype(np.float64)
    writer = gguf.GGUFWriter(path, "qwen35")
    tensors = []
    max_relative_component = 0.0
    try:
        for name, field in reader.fields.items():
            if name.startswith(("GGUF.", PREFIX)) or name == "general.architecture":
                continue
            subtype = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), field.types[0], sub_type=subtype)
        if not folded:
            if malformed == "wrong_version_type":
                writer.add_int32(PREFIX + "version", 1)
            elif malformed != "missing_version":
                writer.add_uint32(PREFIX + "version", 2 if malformed == "unsupported_version" else 1)
            writer.add_string(PREFIX + "kind", "residual_direction_removal")
            if malformed == "wrong_array_type":
                writer.add_string(PREFIX + "left_weights", "invalid")
            else:
                writer.add_key_value(PREFIX + "left_weights", left, gguf.GGUFValueType.ARRAY,
                                     sub_type=gguf.GGUFValueType.STRING)
            writer.add_key_value(PREFIX + "right_weights", right, gguf.GGUFValueType.ARRAY,
                                 sub_type=gguf.GGUFValueType.STRING)
        for tensor in reader.tensors:
            if tensor.name == DIRECTION or (tied and tensor.name in ("output.weight", "output.weight.spin8_bank")):
                continue
            data, kind = tensor.data, tensor.tensor_type
            if folded and (tensor.name in left or tensor.name in right):
                if kind != gguf.GGMLQuantizationType.F32:
                    raise ValueError("Folded control must use F32 decoded matrices")
                matrix = data.astype(np.float64)
                if tensor.name in left:
                    matrix -= unit[:, None] * (unit @ matrix)[None, :]
                    component = unit @ matrix
                else:
                    matrix -= (matrix @ unit)[:, None] * unit[None, :]
                    component = matrix @ unit
                max_relative_component = max(max_relative_component,
                    float(np.linalg.norm(component) / max(np.linalg.norm(matrix), 1e-30)))
                data = matrix.astype("<f4")
            writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_dtype=kind)
            tensors.append(data)
        if not folded and malformed != "missing_direction":
            if malformed == "wrong_direction_type":
                stored = stored.astype("<f8")
            writer.add_tensor_info(DIRECTION, stored.shape, stored.dtype, stored.nbytes)
            tensors.append(stored)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for data in tensors:
            writer.write_tensor_data(data)
    finally:
        writer.close()
    return {"max_relative_direction_component_of_folded_weights": max_relative_component}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--float-fixture", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.float_fixture = args.float_fixture.resolve(strict=True)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.float_fixture.stat().st_size > 32*1024*1024:
        parser.error("Use a tiny float fixture of at most 32 MiB")
    if any(args.out_dir.iterdir()):
        parser.error("Use a new empty output directory")
    report = {"schema": "spin8-native-projection-test-v1", "status": "FAIL",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic-direction linear algebra and tiny Qwen35 graph; no behavioral claim",
              "runtime_executable": artifact(args.binary), "source": artifact(args.float_fixture),
              "parity": [], "loader_rejections": []}
    try:
        source = bridge.open_source(args.float_fixture, native=False)
        hidden = int(source.fields["qwen35.embedding_length"].contents())
        direction = np.random.default_rng(2048).normal(size=hidden).astype("<f4")
        np.save(args.out_dir / "synthetic-direction.npy", direction)
        edited = args.out_dir / "edited-f32.gguf"
        native = args.out_dir / "edited-spin8.gguf"
        decoded = args.out_dir / "edited-decoded.gguf"
        run([sys.executable, str(HERE / "edit_weights.py"), str(args.float_fixture), str(edited),
             "--remove-direction", str(args.out_dir / "synthetic-direction.npy"), "--outtype", "f32"],
            args.out_dir / "edit.log")
        run([sys.executable, str(HERE / "convert.py"), "quantize", str(edited), str(native),
             "--decoded-control", str(decoded), "--include-embedding", "--scale-steps", "5"],
            args.out_dir / "quantize.log")
        actual_reader = bridge.open_source(native, native=True)
        control_reader = bridge.open_source(decoded, native=False)
        left = [t.name for t in actual_reader.tensors
                if re.fullmatch(r"blk\.\d+\.(attn_output|ffn_down|ssm_out)\.weight", t.name)]
        right = ["token_embd.weight"]
        direction = next(t.data.copy() for t in actual_reader.tensors if t.name == DIRECTION)
        prefix = args.out_dir / "no-alloc"
        run([str(args.binary), "--model", str(native), "--output-prefix", str(prefix),
             "--ngl", "0", "--no-alloc", "--no-mmap"], args.out_dir / "no-alloc-console.log")
        no_alloc = json.loads(Path(str(prefix) + ".json").read_text())
        no_alloc_log = Path(str(prefix) + ".log").read_text()
        if not (no_alloc.get("no_alloc_graph_created") and "load_tensors_spin8_projection:" in no_alloc_log):
            raise RuntimeError("Memory-fit graph did not report active projection bindings")
        report["no_alloc"] = {"status": no_alloc["status"], "graph_created": True,
                              "tensor_execution": False, "report": str(prefix) + ".json"}
        variants = [("both", left, right, False, 3.75),
                    ("left_only", left, [], False, 1.0),
                    ("right_only", [], right, False, 1.0),
                    ("tied_head", left, right, True, 1.0),
                    ("small_direction", left, right, False, 1e-30)]
        for name, ls, rs, tied, scale in variants:
            variant = args.out_dir / (name + ".gguf")
            control = args.out_dir / (name + "-folded-f32.gguf")
            scaled = direction * np.float32(scale)
            write_variant(actual_reader, variant, scaled, ls, rs, tied=tied)
            algebra = write_variant(control_reader, control, scaled, ls, rs, folded=True, tied=tied)
            target = args.out_dir / name
            command = [sys.executable, str(HERE / "verify_native.py"), "--binary", str(args.binary),
                       "--model", str(variant), "--control", str(control), "--out-dir", str(target),
                       "--ngl", "0", "--control-ngl", "0", "--threads", "2",
                       "--mmap" if name in ("both", "tied_head") else "--no-mmap"]
            run(command, args.out_dir / (name + "-parity.log"))
            parity = json.loads((target / "parity.json").read_text())
            report["parity"].append({"case": name, "status": parity["status"],
                                    "comparison": parity.get("comparison"), "report": str(target / "parity.json"),
                                    **algebra})
        for case in NEGATIVE_CASES:
            variant = args.out_dir / ("bad-" + case + ".gguf")
            write_variant(actual_reader, variant, direction, left, right, malformed=case)
            runs = [invoke_loader(args.binary, variant, args.out_dir / (case + "-" + mode), mapped, True)
                    for mapped, mode in ((False, "no-mmap"), (True, "mmap"))]
            report["loader_rejections"].append({"case": case, "runs": runs,
                "status": "PASS" if all(r["status"] == "PASS" for r in runs) else "FAIL"})
        report["source_unchanged"] = artifact(args.float_fixture) == report["source"]
        report["status"] = "PASS" if (report["source_unchanged"] and
            all(r["status"] == "PASS" for r in report["parity"] + report["loader_rejections"])) else "FAIL"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    output = args.out_dir / "projection-tests.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "parity_cases": len(report["parity"]),
                      "negative_cases": len(report["loader_rejections"]), "report": str(output),
                      "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
