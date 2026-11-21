#!/usr/bin/env python3
"""Exercise clean native-loader rejection of malformed tiny Spin8 fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import convert as bridge
import gguf
from verify_native import artifact


CASES = (
    "missing_codec_version",
    "wrong_codec_version_type",
    "unsupported_codec_version",
    "missing_bank",
    "wrong_bank_shape",
    "wrong_bank_type",
    "nonorthogonal_bank",
    "invalid_scale",
    "reserved_code",
    "truncated_payload",
)


def write_variant(reader: gguf.GGUFReader, path: Path, case: str) -> None:
    weight = next(tensor.name for tensor in reader.tensors if tensor.tensor_type == bridge.TYPE)
    bank_name = weight + bridge.BANK_SUFFIX
    writer = gguf.GGUFWriter(path, "qwen35")
    try:
        for name, field in reader.fields.items():
            if name.startswith("GGUF.") or name == "general.architecture":
                continue
            if name == "spin8.codec_version":
                if case == "missing_codec_version":
                    continue
                if case == "wrong_codec_version_type":
                    writer.add_int32(name, 1)
                    continue
                if case == "unsupported_codec_version":
                    writer.add_uint32(name, 2)
                    continue
            subtype = field.types[1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), field.types[0], sub_type=subtype)
        writer.add_string("spin8.fixture.negative_case", case)
        tensors = []
        for tensor in reader.tensors:
            if tensor.name == bank_name and case == "missing_bank":
                continue
            data, kind = tensor.data, tensor.tensor_type
            if tensor.name == bank_name:
                if case == "wrong_bank_shape":
                    data = data[:8].copy()
                elif case == "wrong_bank_type":
                    data, kind = data.astype("<f4"), gguf.GGMLQuantizationType.F32
                elif case == "nonorthogonal_bank":
                    data = data.copy()
                    data.flat[0] += 0.25
            if tensor.name == weight and case in ("invalid_scale", "reserved_code"):
                data = data.copy()
                raw = data.reshape(-1).view("u1")
                if case == "invalid_scale":
                    raw[32:34] = [0x00, 0x7E]  # FP16 NaN.
                else:
                    raw[:32] = 0
                    raw[:2] = [0xFF, 0xFF]  # Reserved ternary code with a positive scale.
                    raw[32:34] = [0x00, 0x3C]
            writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_dtype=kind)
            tensors.append(data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for data in tensors:
            writer.write_tensor_data(data)
    finally:
        writer.close()
    if case == "truncated_payload":
        with path.open("r+b") as handle:
            handle.truncate(path.stat().st_size - 64)


def invoke(binary: Path, model: Path, prefix: Path, mapped: bool, expect_failure: bool) -> dict:
    command = [
        str(binary), "--model", str(model), "--output-prefix", str(prefix), "--ngl", "0",
        "--threads", "2", "--mmap" if mapped else "--no-mmap",
    ]
    console = Path(str(prefix) + ".console.log")
    try:
        with console.open("wb") as output:
            process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False, timeout=60)
        metadata_path = Path(str(prefix) + ".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if expect_failure:
            passed = (
                process.returncode == 1
                and metadata.get("status") == "FAIL"
                and "model loading failed" in metadata.get("error", "")
            )
        else:
            passed = process.returncode == 0 and metadata.get("status") == "PASS" and metadata.get("finite_logits") is True
        return {
            "status": "PASS" if passed else "FAIL", "returncode": process.returncode,
            "terminated_by_signal": process.returncode < 0,
            "load_mode": "mmap" if mapped else "no_mmap", "command": command,
            "native_error": metadata.get("error"), "console_log": str(console),
            "native_report": str(metadata_path) if metadata_path.exists() else None,
        }
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "error": "timeout", "load_mode": "mmap" if mapped else "no_mmap", "console_log": str(console)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.binary, args.fixture = args.binary.resolve(strict=True), args.fixture.resolve(strict=True)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fixture.stat().st_size > 32 * 1024 * 1024:
        parser.error("Use a tiny fixture of at most 32 MiB; this test writes malformed copies")
    reader = bridge.open_source(args.fixture, native=True)
    if not any(tensor.tensor_type == bridge.TYPE for tensor in reader.tensors):
        parser.error("fixture has no native Spin8 tensors")
    report = {
        "schema": "spin8-native-loader-rejections-v1", "status": "FAIL",
        "scope": "tiny-model loader validation; every malformed input must fail without a signal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_executable": artifact(args.binary), "source_fixture": artifact(args.fixture),
        "baseline": [], "cases": [],
    }
    for mapped in (False, True):
        mode = "mmap" if mapped else "no_mmap"
        report["baseline"].append(invoke(args.binary, args.fixture, args.out_dir / ("valid_" + mode), mapped, False))
    if all(item["status"] == "PASS" for item in report["baseline"]):
        for case in CASES:
            model = args.out_dir / (case + ".gguf")
            if model.resolve() == args.fixture:
                raise ValueError("test output must not overwrite source fixture")
            write_variant(reader, model, case)
            item = {"case": case, "model": artifact(model), "runs": []}
            for mapped in (False, True):
                mode = "mmap" if mapped else "no_mmap"
                item["runs"].append(invoke(args.binary, model, args.out_dir / (case + "_" + mode), mapped, True))
            item["status"] = "PASS" if all(run["status"] == "PASS" for run in item["runs"]) else "FAIL"
            report["cases"].append(item)
    passed = len(report["cases"]) == len(CASES) and all(item["status"] == "PASS" for item in report["cases"])
    report["status"] = "PASS" if passed else "FAIL"
    report["cases_passed"] = sum(item["status"] == "PASS" for item in report["cases"])
    report["source_unchanged"] = artifact(args.fixture) == report["source_fixture"]
    if not report["source_unchanged"]:
        report["status"] = "FAIL"
    path = args.out_dir / "loader-rejections.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases_passed": report["cases_passed"], "cases_total": len(CASES), "report": str(path)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
