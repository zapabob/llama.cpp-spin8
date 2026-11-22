#!/usr/bin/env python3
"""Render the ordinary Qwen3.8 Spin8 release card from matching final evidence.

Run after pipeline.release_files has prepared provenance.json. This helper does
not import or modify the active payload. It stages only the README and small
evidence/licence files; add its printed upload paths to the release file list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import os

SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
BASE_COMMIT = "7fce5439fe451cd48340b112ec37d2befe02c776"
PRECISION_KEYS = ("GGML_CUDA_SPIN8_VALIDATE_F32", "NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")
MAX_REPORT_BYTES = 64 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bytes(path: Path) -> bytes:
    require(path.is_file() and not path.is_symlink(), "Missing regular file: " + str(path))
    require(path.stat().st_size <= MAX_REPORT_BYTES, "Report exceeds size limit: " + str(path))
    return path.read_bytes()


def read_json(path: Path) -> dict:
    value = json.loads(read_bytes(path))
    require(isinstance(value, dict), "Expected a JSON object: " + str(path))
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_artifact(path: Path, record: dict):
    require(path.is_file() and not path.is_symlink(), "Missing artifact: " + str(path))
    require(path.stat().st_size == record.get("bytes") and sha256(path) == record.get("sha256"),
            "Artifact differs from its report: " + str(path))


def runtime_identity(build: dict) -> dict:
    return {"base_commit": build["base_commit"], "patch_sha256": build["patch_sha256"],
            "device": build["device_requested"],
            "binaries": {name: build[name]["sha256"] for name in ("llama-server", "test-spin8-native")},
            "libraries": {name: item["sha256"] for name, item in build.get("libraries", {}).items()}}


def optional_benchmark(path: Path | None, model_sha: str, server_sha: str, copies: dict) -> str:
    if path is None:
        return ("No completed quality benchmark is included in this card. Mathematical answer accuracy, "
                "perplexity and changes relative to the source model have not been established.")
    report = read_json(path)
    require(report.get("schema") == "spin8-gsm8k-benchmark-v1" and report.get("status") == "COMPLETE",
            "Supplied benchmark is not complete")
    inputs, dataset, protocol = report["inputs"], report["dataset"], report["protocol"]
    require(inputs["candidate"]["sha256"] == model_sha and inputs["server"]["sha256"] == server_sha,
            "Benchmark model/runtime differs from this release")
    require(dataset.get("repo") == "openai/gsm8k" and re.fullmatch(r"[0-9a-f]{40}", dataset.get("revision", ""))
            and valid_hash(dataset.get("file_sha256")), "Benchmark dataset is not pinned")
    requested = protocol.get("samples")
    require(type(requested) is int and requested > 0, "Invalid benchmark sample count")
    for label, model in report["models"].items():
        summary, items = model["summary"], model["items"]
        require(label in inputs and model.get("status") == "COMPLETE" and valid_hash(inputs[label].get("sha256")),
                "Incomplete benchmark model")
        require(summary.get("requested_items") == requested and summary.get("completed_items") == requested
                and len(items) == requested and all(type(item.get("correct")) is bool for item in items),
                "Benchmark sample coverage is incomplete")
        require(summary.get("correct") == sum(item["correct"] for item in items)
                and math.isclose(summary["accuracy"], summary["correct"] / requested, rel_tol=0, abs_tol=1e-12),
                "Benchmark accuracy is inconsistent")
    candidate = report["models"]["candidate"]["summary"]
    markdown = path.with_suffix(".md")
    require(hashlib.sha256(read_bytes(markdown)).hexdigest() == report["artifacts"]["benchmark_md_sha256"],
            "Benchmark Markdown differs from its completed report")
    copies["benchmarks/benchmark.json"] = read_bytes(path)
    copies["benchmarks/benchmark.md"] = read_bytes(markdown)
    problem_label = "problem" if requested == 1 else "problems"
    return (f"A matching GSM8K run completed **{requested:,} {problem_label}**, with **{candidate['correct']:,} correct "
            f"({100 * candidate['accuracy']:.2f}%)** under its recorded exact-answer scoring protocol. "
            f"The dataset revision is `{dataset['revision']}`.\n\n"
            "See [the complete benchmark report](benchmarks/benchmark.md) and [per-item evidence](benchmarks/benchmark.json) "
            "for sample selection, prompts, token budgets, answer extraction, uncertainty and any recorded baseline. "
            "This card reports the candidate measurement only; it does not infer a capability improvement from it.")


def prepare(args) -> tuple[str, dict[str, bytes]]:
    work, model = args.work_dir.resolve(), args.gguf.resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", model.name), "Unsafe GGUF filename")
    conversion_path = model.with_name(model.name + ".conversion.json")
    verification_path = model.with_name(model.name + ".verification.json")
    conversion, verification = read_json(conversion_path), read_json(verification_path)
    build_path = work / "reports/build-report.json"
    build = read_json(build_path)
    require(build.get("schema") == "spin8-native-build-v2" and build.get("status") == "PASS"
            and build.get("base_commit") == BASE_COMMIT and build.get("device_requested") == "cuda"
            and build.get("cuda_device_execution") is True and build.get("projected_native_parity") == "PASS",
            "A completed CUDA build with projected parity is required")
    require(build.get("source", {}).get("verified") is True and valid_hash(build.get("patch_sha256")),
            "Build source/patch identity is incomplete")
    required_stages = {"CPU codec correctness", "CUDA codec actual device correctness", "native-parity",
                       "projected-native-parity", "server generation APIs"}
    require(required_stages <= {entry["name"] for entry in build.get("stages", [])}, "Mandatory build stages are missing")
    for entry in build.get("stages", []):
        require(entry.get("status") == "PASS", "Build contains an incomplete stage")
    for name in ("llama-server", "test-spin8-native"):
        check_artifact(Path(build[name]["path"]), build[name])
    require(conversion.get("schema") == "spin8-colab-pipeline-v2" and conversion.get("status") == "COMPLETE"
            and verification.get("schema") == "spin8-colab-pipeline-v2" and verification.get("status") == "PASS",
            "Conversion and real-model verification must be complete")
    identity, result, output = conversion["identity"], conversion["conversion"], conversion["output"]
    require(conversion["config_fingerprint"] == digest_json(identity)
            and verification.get("config_fingerprint") == conversion["config_fingerprint"], "Configuration identity differs")
    require(identity["runtime"] == verification.get("runtime") == runtime_identity(build), "Runtime identity differs")
    require(verification.get("output") == output and verification.get("device") == "cuda", "Verification artifact/device differs")
    require(result.get("status") == "converted" and result.get("output_sha256") == output["sha256"]
            and result.get("file_bytes") == output["bytes"], "Converter output identity differs")
    check_artifact(model, output)
    expected = {"math_alpha": 0.0, "direction_sha256": None, "seed": 81237, "scale_steps": 17,
                "chunk_groups": 8192, "tile_elements": 1048576, "encoder": "torch", "device": "cuda",
                "compute_dtype": "float32", "bank_kind": "triality", "no_mtp": True}
    require(all(identity.get(key) == value for key, value in expected.items()), "This template requires the recorded ordinary baseline settings")
    provenance = result["provenance"]
    require(provenance.get("optimizer_steps") == 0 and provenance.get("answer_data_used") is False
            and provenance.get("projection_runtime_required") is False and provenance.get("math_alpha") is None
            and provenance.get("donor_revision") is None and provenance.get("edited_tensors") == [], "Unexpected edits or training")
    require(provenance.get("source_revision") == SOURCE_REVISION and provenance.get("no_mtp") is True
            and provenance.get("encoder") == "torch" and provenance.get("encoder_device") == "cuda"
            and provenance.get("compute_dtype") == "float32", "Converter provenance settings differ")
    encoded = result["memory"]["encoded_tensors_this_run"]
    require(type(encoded) is int and encoded >= 0
            and encoded + result["resume"]["hits"] == result["quantized_tensor_count"]
            and provenance.get("cuda_preflight_executed") is True and
            ((encoded > 0 and provenance.get("encoder_cuda_executed") is True)
             or (encoded == 0 and result["resume"]["hits"] == result["quantized_tensor_count"])),
            "Neither CUDA encoding nor a complete matching CUDA-cache resume was recorded")
    source = read_json(args.source_manifest)
    require(source.get("schema") == "spin8-hf-source-v2" and source.get("status") == "COMPLETE"
            and source.get("weight_payload_downloaded") is True and source.get("repo") == SOURCE_REPO
            and source.get("revision") == SOURCE_REVISION, "Source manifest is incomplete or uses another checkpoint")
    require(identity["sources"] == [{"role": "source", "repo": SOURCE_REPO, "revision": SOURCE_REVISION,
                                      "content_fingerprint": source["content_fingerprint"]}], "Conversion source identity differs")
    files = source["files"]
    require(files and len({entry["path"] for entry in files}) == len(files)
            and all(entry.get("verified") is True and valid_hash(entry.get("sha256")) for entry in files),
            "Source inventory is unverified or duplicated")
    fingerprint = hashlib.sha256(json.dumps({"repo": source["repo"], "revision": source["revision"],
        "files": [{key: entry[key] for key in ("path", "size", "sha256")} for entry in files]}, sort_keys=True).encode()).hexdigest()
    require(fingerprint == source["content_fingerprint"], "Source manifest fingerprint is inconsistent")
    shards = [entry for entry in files if entry["path"].endswith(".safetensors")]
    planned = source["planned_weight_files"]
    require(shards and {entry["path"] for entry in shards} == {entry["path"] for entry in planned}
            and sum(entry["size"] for entry in shards) == source["planned_weight_file_bytes"]
            and sum(entry["payload_bytes"] for entry in source["safetensors_headers"]) == source["verified_payload_bytes"],
            "Source shard inventory/totals are inconsistent")
    copies = {"conversion.json": read_bytes(conversion_path), "verification.json": read_bytes(verification_path),
              "evidence/build-report.json": read_bytes(build_path), "evidence/source.json": read_bytes(args.source_manifest)}
    licence_names = []
    for entry in files:
        name = entry["path"]
        if name in ("README.md", "LICENSE", "LICENSE.txt", "NOTICE"):
            data = read_bytes(args.source_manifest.parent / name)
            require(len(data) == entry["size"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                    "Upstream attribution/licence file differs: " + name)
            copies["upstream/source/" + name] = data
            if name.startswith("LICENSE"):
                require(b"Apache License" in data and b"Version 2.0" in data, "Unexpected upstream licence text")
                licence_names.append(name)
    require(licence_names and "upstream/source/README.md" in copies, "Source licence and model card are required")
    environment_path = work / "reports/environment.json"
    environment = read_json(environment_path)
    gpu, torch = environment["gpu"], environment["torch"]
    require(gpu.get("torch_device_executed") is True and "A100" in gpu.get("name", "")
            and torch["version"].split("+")[0] == identity["versions"]["torch"].split("+")[0],
            "GPU/environment evidence differs")
    copies["evidence/environment.json"] = read_bytes(environment_path)
    profile = build["parity_precision_profile"]
    require(profile.get("cache_type_k") == profile.get("cache_type_v") == "f32"
            and profile.get("atol") == profile.get("rtol") == 1e-4
            and profile.get("server_smoke_uses_parity_environment") is False, "Unexpected parity precision profile")
    precision = profile["precision_environment"]
    require(set(precision) == set(PRECISION_KEYS) and precision[PRECISION_KEYS[0]] == "1"
            and precision[PRECISION_KEYS[1]] == "0", "Strict parity environment is missing")
    rows = []
    for stage, label in (("native-parity", "Ordinary Spin8"), ("projected-native-parity", "Projected fixture")):
        path = work / "reports" / stage / "parity.json"
        parity = read_json(path)
        comparison = parity["comparison"]
        require(parity.get("status") == comparison.get("status") == "PASS"
                and comparison.get("tolerance_violations") == 0
                and comparison.get("all_observed_required_op_output_devices_match") is True
                and comparison.get("cache_type_k") == comparison.get("cache_type_v") == "f32"
                and comparison.get("atol") == comparison.get("rtol") == 1e-4
                and parity.get("precision_environment") == precision
                and parity["runtime_executable"]["sha256"] == build["test-spin8-native"]["sha256"],
                "Parity report does not match the verified runtime/profile: " + stage)
        rows.append(f"| {label} | {comparison['elements']:,} | {comparison['max_absolute_error']:.9g} | {comparison['rmse']:.9g} | 0 |")
        copies["evidence/" + stage + ".json"] = read_bytes(path)
    native = verification["native"]
    api = verification["api"]
    require(native.get("finite_logits") is True and native.get("actual_device_verified") is True
            and native.get("required_device") == "cuda" and native.get("required_operation") == "MUL_MAT_SPIN8"
            and api.get("status") == "PASS" and api["model"]["sha256"] == output["sha256"]
            and api["binary"]["sha256"] == build["llama-server"]["sha256"]
            and api.get("native_completion", {}).get("tokens_predicted", 0) > 0
            and api.get("openai_completion", {}).get("choices"), "Real-model execution/API evidence is incomplete")
    metadata_path = Path(verification["native_metadata"]["path"])
    check_artifact(metadata_path, verification["native_metadata"])
    metadata = read_json(metadata_path)
    require(metadata.get("cache_type_k") == metadata.get("cache_type_v") == "f16"
            and metadata.get("status") == "PASS" and metadata.get("finite_logits") is True
            and type(metadata.get("n_ctx_requested")) is int and metadata["n_ctx_requested"] > 0
            and Path(metadata["model"]).resolve() == model
            and math.prod(metadata["logits_shape"]) == native["logit_elements"], "Real-model native profile differs")
    check_artifact(Path(native["logits"]["path"]), native["logits"])
    copies["evidence/native-execution.json"] = read_bytes(metadata_path)
    records = result["tensors"]
    require(records and result["quantized_tensor_count"] == len(records)
            and len({entry["name"] for entry in records}) == len(records) and result.get("payload_bpw") == 2.125,
            "Compressed tensor inventory is inconsistent")
    require([entry["name"] for entry in records] == provenance.get("quantized_tensors")
            and "token_embd.weight" in provenance["quantized_tensors"], "Compressed inventory/embedding provenance differs")
    weights = packed = 0
    for entry in records:
        shape = entry["shape"]
        require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape) and shape[1] % 128 == 0,
                "Invalid compressed matrix shape")
        count = math.prod(shape)
        require(entry["payload_bytes"] == count // 128 * 34, "Compressed matrix byte count differs")
        weights += count
        packed += entry["payload_bytes"]
    with model.open("rb") as handle:
        magic, version, tensor_count, _ = struct.unpack("<4sIQQ", handle.read(24))
    canonical_count = result["memory"]["canonical_tensor_count"]
    require(magic == b"GGUF" and version == 3 and tensor_count == canonical_count + len(records),
            "GGUF header/tensor count differs from the completed conversion")
    release_provenance = read_json(args.release_dir / "provenance.json")
    require(release_provenance.get("output") == output and release_provenance.get("identity") == identity
            and release_provenance.get("verification") == verification
            and release_provenance.get("build") == build, "Prepare matching release provenance before rendering")
    benchmark_path = args.benchmark
    recorded_benchmark = release_provenance.get("benchmark")
    if recorded_benchmark:
        artifact = recorded_benchmark["artifacts"]["benchmarks/benchmark.json"]
        benchmark_path = benchmark_path or Path(artifact["path"])
        check_artifact(benchmark_path, artifact)
    else:
        require(benchmark_path is None, "Prepare matching release provenance before adding a benchmark")
    values = {
        "SOURCE_REVISION": SOURCE_REVISION, "MODEL_FILE": model.name, "MODEL_BYTES": f"{output['bytes']:,}",
        "MODEL_BYTES_RAW": str(output["bytes"]), "MODEL_GIB": f"{output['bytes'] / 2**30:.3f}",
        "MODEL_SHA256": output["sha256"], "CONVERSION_COMPLETED": conversion["completed_utc"],
        "VERIFICATION_COMPLETED": verification["completed_utc"], "SOURCE_FINGERPRINT": fingerprint,
        "CONFIG_FINGERPRINT": conversion["config_fingerprint"], "BASE_COMMIT": BASE_COMMIT,
        "PATCH_SHA256": build["patch_sha256"], "GPU_NAME": gpu["name"],
        "GPU_GIB": f"{gpu['total_memory_bytes'] / 2**30:.2f}", "TORCH_VERSION": identity["versions"]["torch"],
        "CUDA_VERSION": torch["cuda_build"], "SHARD_COUNT": str(len(shards)),
        "SOURCE_FILE_BYTES": f"{sum(entry['size'] for entry in shards):,}",
        "SOURCE_PAYLOAD_BYTES": f"{source['verified_payload_bytes']:,}",
        "CANONICAL_TENSORS": f"{canonical_count:,}", "QUANTISED_TENSORS": f"{len(records):,}",
        "QUANTISED_WEIGHTS": f"{weights:,}", "PACKED_BYTES": f"{packed:,}",
        "BANK_BYTES": f"{4608 * len(records):,}", "GGUF_TENSORS": f"{tensor_count:,}",
        "PARITY_ROWS": "\n".join(rows), "PRECISION_PROFILE": json.dumps(profile, indent=2, allow_nan=False),
        "LOGIT_ELEMENTS": f"{native['logit_elements']:,}", "NATIVE_CONTEXT": str(metadata["n_ctx_requested"]),
        "LICENCE_LINKS": "\n".join(f"- [{name}](upstream/source/{name})" for name in licence_names),
        "BENCHMARK_SECTION": optional_benchmark(benchmark_path, output["sha256"], build["llama-server"]["sha256"], copies),
    }
    template = read_bytes(args.template).decode("utf-8")
    require(set(re.findall(r"@([A-Z_0-9]+)@", template)) == set(values), "Template placeholders differ from renderer fields")
    card = re.sub(r"@([A-Z_0-9]+)@", lambda match: str(values[match[1]]), template)
    return card, copies


def render(args) -> dict:
    destination = args.release_dir.resolve()
    for name in ("payload", "llama.cpp", "sources", "outputs"):
        require(not destination.is_relative_to(args.work_dir.resolve() / name), "Release directory overlaps live pipeline files")
    card, copies = prepare(args)
    copies["README.md"] = card.encode("utf-8")
    for name, data in copies.items():
        path = destination / name
        require(not path.is_symlink() and path.resolve().is_relative_to(destination), "Unsafe release output path")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"status": "RENDERED", "readme": str(destination / "README.md"),
            "additional_upload_files": [{"path_in_repo": name, "local_path": str(destination / name)}
                                        for name in sorted(copies)]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[2] / "docs/spin8/HF_MODEL_CARD_TEMPLATE.md")
    args = parser.parse_args()
    try:
        print(json.dumps(render(args), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, "Model card was not rendered: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
