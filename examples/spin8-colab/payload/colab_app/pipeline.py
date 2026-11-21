#!/usr/bin/env python3
"""Pinned HF -> direct weight edits -> private Spin8 GGUF -> verified HF release."""
from __future__ import annotations

import argparse
import array
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colab_app import bootstrap, hf_fetch

SCHEMA = "spin8-colab-pipeline-v2"
GIB = 1 << 30
DEFAULTS = {
    "source_repo": hf_fetch.QWEN_REPO, "source_revision": "main", "work_dir": "/content/spin8-work",
    "output_name": "Qwen3.8-27B-Spin8.gguf", "device": "cuda", "math_delta": False,
    "math_alpha": 0.25, "direction_path": "", "seed": 81237, "scale_steps": 17,
    "chunk_groups": 256, "tile_elements": 1048576, "hf_repo": "", "hf_private": True,
    "payload_dir": "", "max_workers": 2, "disk_reserve_gib": 8,
    "max_materialized_bytes": 2 << 30,
}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path: Path | str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The pipeline config must be a JSON object")
    if "revision" in value:
        if "source_revision" in value and value["source_revision"] != value["revision"]:
            raise ValueError("revision and source_revision disagree")
        value["source_revision"] = value.pop("revision")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown config keys: " + ", ".join(sorted(unknown)) + "; store HF_TOKEN only in the environment")
    config = {**DEFAULTS, **value}
    for key in ("source_repo", "hf_repo"):
        if (key == "hf_repo" and config[key] == ""):
            continue
        if not isinstance(config[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config[key]):
            raise ValueError(key + " must be an exact owner/repository ID")
    for key in ("math_delta", "hf_private"):
        if type(config[key]) is not bool:
            raise ValueError(key + " must be a JSON boolean")
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not isinstance(config["source_revision"], str) or not config["source_revision"].strip():
        raise ValueError("source_revision is required")
    name = config["output_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", name):
        raise ValueError("output_name must be a simple .gguf filename without directories")
    for key in ("math_alpha", "disk_reserve_gib"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(key + " must be a finite number")
    if not 0 <= config["math_alpha"] <= 1 or config["disk_reserve_gib"] < 1:
        raise ValueError("math_alpha must be in [0,1] and disk_reserve_gib must be at least 1")
    for key, minimum in (("seed", 0), ("scale_steps", 1), ("chunk_groups", 1),
                         ("tile_elements", 128), ("max_materialized_bytes", 128), ("max_workers", 1)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["max_workers"] > 8:
        raise ValueError("max_workers must be <= 8")
    if config["math_delta"] and config["source_repo"] != hf_fetch.QWEN_REPO:
        raise ValueError("The audited math donor is tied to Qwen/Qwen3.8-27B")
    for key in ("work_dir", "payload_dir", "direction_path"):
        if not isinstance(config[key], str):
            raise ValueError(key + " must be a path string")
    config["work_dir"] = str(Path(config["work_dir"]).expanduser().resolve())
    config["payload_dir"] = str(Path(config["payload_dir"] or Path(config["work_dir"]) / "payload").expanduser().resolve())
    if config["direction_path"]:
        config["direction_path"] = str(Path(config["direction_path"]).expanduser().resolve())
    return config


def paths(config: dict) -> dict[str, Path]:
    work = Path(config["work_dir"])
    output = work / "outputs" / config["output_name"]
    return {"work": work, "output": output,
            "conversion": output.with_name(output.name + ".conversion.json"),
            "verification": output.with_name(output.name + ".verification.json"),
            "reports": work / "reports/pipeline", "repo": work / "llama.cpp"}


def source_specs(config: dict) -> list[dict]:
    root = Path(config["work_dir"]) / "sources"
    if config["math_delta"]:
        return [
            {"role": "source", "repo": hf_fetch.QWEN_REPO, "revision": hf_fetch.MATH_BASE_REVISION,
             "directory": root / "math" / ("base-" + hf_fetch.MATH_BASE_REVISION[:12])},
            {"role": "donor", "repo": hf_fetch.MATH_DONOR_REPO, "revision": hf_fetch.MATH_DONOR_REVISION,
             "directory": root / "math" / ("donor-" + hf_fetch.MATH_DONOR_REVISION[:12])},
        ]
    key = config["source_repo"].replace("/", "--") + "-" + digest_json(config["source_revision"])[:12]
    return [{"role": "source", "repo": config["source_repo"], "revision": config["source_revision"],
             "directory": root / "plain" / key}]


def read_source(spec: dict, *, complete: bool = True) -> dict:
    directory = spec["directory"]
    path = directory / "source.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Run download first; source manifest is missing: " + str(path))
    report = json.loads(path.read_text())
    if report.get("schema") != hf_fetch.SCHEMA or report.get("repo") != spec["repo"]:
        raise RuntimeError("Source manifest repository/schema differs from the requested source")
    if not re.fullmatch(r"[0-9a-f]{40}", report.get("revision", "")):
        raise RuntimeError("Source manifest has no immutable revision")
    if spec["revision"] not in (report.get("requested_revision"), report.get("revision")):
        raise RuntimeError("Source manifest revision differs from the requested source")
    if complete and (report.get("status") != "COMPLETE" or report.get("weight_payload_downloaded") is not True):
        raise RuntimeError("Source is not COMPLETE; finish download before conversion")
    if not complete and report.get("status") not in {"METADATA_ONLY", "COMPLETE"}:
        raise RuntimeError("Source metadata is not complete")
    records = report.get("files", [])
    if not records or len({record["path"] for record in records}) != len(records):
        raise RuntimeError("Source manifest file inventory is empty or duplicated")
    for record in records:
        name = hf_fetch._safe_relative(record["path"])
        file = directory / name
        if (not file.is_file() or file.is_symlink() or not file.resolve().is_relative_to(directory.resolve())
                or file.stat().st_size != record["size"] or record.get("verified") is not True
                or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise RuntimeError("Source file is missing/unverified or changed size: " + name)
    expected = hashlib.sha256(json.dumps({"repo": report["repo"], "revision": report["revision"],
                 "files": [{"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
                           for item in records]}, sort_keys=True).encode()).hexdigest()
    if expected != report.get("content_fingerprint"):
        raise RuntimeError("Source manifest content fingerprint is invalid")
    if complete:
        planned = {item["path"] for item in report.get("planned_weight_files", [])}
        if not planned or not planned <= {item["path"] for item in records}:
            raise RuntimeError("COMPLETE manifest does not cover all planned shards")
    return report


def _metadata(config: dict) -> list[tuple[dict, dict]]:
    result = []
    for spec in source_specs(config):
        existing = spec["directory"] / "source.json"
        # Planning after a download must not downgrade COMPLETE to METADATA_ONLY.
        if existing.is_file() and json.loads(existing.read_text()).get("status") == "COMPLETE":
            report = read_source(spec)
        else:
            report = hf_fetch.fetch_source(spec["directory"], spec["repo"], spec["revision"],
                        token=os.environ.get("HF_TOKEN"), metadata_only=True, max_workers=config["max_workers"])
        result.append((spec, report))
    return result


def disk_plan(config: dict, sources: list[tuple[dict, dict]]) -> dict:
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    source_bytes = missing_bytes = largest_shard = 0
    for spec, report in sources:
        for item in report["planned_weight_files"]:
            size = item["size"]
            source_bytes += size
            largest_shard = max(largest_shard, size)
            file = spec["directory"] / item["path"]
            present = file.stat().st_size if file.is_file() and not file.is_symlink() else 0
            # A wrong-sized existing file may need a complete temporary replacement.
            missing_bytes += 0 if present == size else size
    primary_bytes = sources[0][1]["planned_weight_file_bytes"]
    packed_cache = math.ceil(primary_bytes * 34 / 256) + (64 << 20)
    # Estimates include float residual tensors, banks, tokenizer and GGUF metadata.
    # Exact sizes are obtained from Safetensors and canonical tensors after download.
    output_estimate = math.ceil(primary_bytes * 0.20) + GIB
    temporary_reserve = largest_shard + GIB
    system_reserve = math.ceil(config["disk_reserve_gib"] * GIB)
    required = missing_bytes + packed_cache + output_estimate + temporary_reserve + system_reserve
    free = shutil.disk_usage(work).free
    return {"source_weight_bytes_total": source_bytes, "source_weight_bytes_missing": missing_bytes,
            "packed_cache_estimate_bytes": packed_cache, "gguf_estimate_bytes": output_estimate,
            "temporary_reserve_bytes": temporary_reserve, "system_reserve_bytes": system_reserve,
            "additional_free_bytes_required_estimate": required, "free_bytes": free,
            "sufficient_estimated_space": free >= required, "full_float_gguf_staged": False,
            "estimate_note": "Pre-download estimate, not a size guarantee; includes both compressed tensor cache and final GGUF. "
                             "Raw source snapshots are retained. No full F16/F32 GGUF is staged."}


def plan(config: dict) -> dict:
    sources = _metadata(config)
    report = {"schema": SCHEMA, "stage": "plan", "status": "READY", "created_utc": timestamp(),
              "math_delta": config["math_delta"], "direction_supplied": bool(config["direction_path"]),
              "sources": [{"role": spec["role"], "repo": item["repo"], "revision": item["revision"],
                           "directory": str(spec["directory"]), "status": item["status"],
                           "planned_weight_file_bytes": item["planned_weight_file_bytes"]}
                          for spec, item in sources], "disk": disk_plan(config, sources),
              "output": str(paths(config)["output"]), "quality_improvement_measured": False}
    bootstrap.atomic_json(paths(config)["reports"] / "plan.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def download(config: dict) -> dict:
    preview = plan(config)
    if not preview["disk"]["sufficient_estimated_space"]:
        disk = preview["disk"]
        raise RuntimeError(f"Insufficient estimated disk space: need {disk['additional_free_bytes_required_estimate']/GIB:.1f} GiB "
                           f"free, have {disk['free_bytes']/GIB:.1f} GiB. Use a larger Colab disk or disable optional math_delta.")
    result = []
    for spec in source_specs(config):
        value = hf_fetch.fetch_source(spec["directory"], spec["repo"], spec["revision"],
                    token=os.environ.get("HF_TOKEN"), max_workers=config["max_workers"])
        read_source(spec)
        result.append({"role": spec["role"], "repo": value["repo"], "revision": value["revision"],
                       "directory": str(spec["directory"]), "content_fingerprint": value["content_fingerprint"],
                       "verified_payload_bytes": value["verified_payload_bytes"]})
    report = {"schema": SCHEMA, "stage": "download", "status": "COMPLETE", "completed_utc": timestamp(),
              "sources": result, "weight_payload_downloaded": True}
    bootstrap.atomic_json(paths(config)["reports"] / "download.json", report)
    return report


def build_gate(config: dict) -> dict:
    path = Path(config["work_dir"]) / "reports/build-report.json"
    if not path.is_file():
        raise RuntimeError("Run bootstrap prepare before conversion/verification")
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or report.get("device_requested") != config["device"]:
        raise RuntimeError("Native runtime build is not PASS for the requested device")
    if config["device"] == "cuda" and report.get("cuda_device_execution") is not True:
        raise RuntimeError("Native build did not execute the mandatory CUDA tests")
    if report.get("projected_native_parity") != "PASS" or report.get("base_commit") != bootstrap.BASE_COMMIT:
        raise RuntimeError("Native build lacks required projected parity or uses another upstream revision")
    repo = paths(config)["repo"]
    if Path(report["source_dir"]).resolve() != repo.resolve():
        raise RuntimeError("Native build belongs to a different work directory")
    for name in ("llama-server", "test-spin8-native"):
        artifact = report[name]
        file = Path(artifact["path"])
        if not file.is_file() or bootstrap.sha256(file) != artifact["sha256"]:
            raise RuntimeError("Native executable changed after bootstrap: " + name)
    for name, artifact in report.get("libraries", {}).items():
        file = Path(artifact["path"])
        if not file.is_file() or bootstrap.sha256(file) != artifact["sha256"]:
            raise RuntimeError("Native library changed after bootstrap: " + name)
    source = report.get("source", {})
    if source.get("verified") is not True or not source.get("patched_files"):
        raise RuntimeError("Build has no verified source fingerprint")
    for name, digest in source["patched_files"].items():
        file = repo / hf_fetch._safe_relative(name)
        if not file.is_file() or bootstrap.sha256(file) != digest:
            raise RuntimeError("Patched source changed after bootstrap: " + name)
    return report


def runtime_identity(build: dict) -> dict:
    return {"base_commit": build["base_commit"], "patch_sha256": build["patch_sha256"],
            "device": build["device_requested"],
            "binaries": {name: build[name]["sha256"] for name in ("llama-server", "test-spin8-native")},
            "libraries": {name: item["sha256"] for name, item in build.get("libraries", {}).items()}}


def conversion_identity(config: dict, build: dict) -> dict:
    sources = [(spec, read_source(spec)) for spec in source_specs(config)]
    direction = config["direction_path"]
    if direction and (not Path(direction).is_file() or Path(direction).is_symlink()):
        raise ValueError("direction_path must point to an existing regular direction file")
    return {"schema": SCHEMA, "sources": [{"role": spec["role"], "repo": report["repo"],
              "revision": report["revision"], "content_fingerprint": report["content_fingerprint"]}
              for spec, report in sources],
            "math_alpha": config["math_alpha"] if config["math_delta"] else 0.0,
            "direction_sha256": bootstrap.sha256(Path(direction)) if direction else None,
            "seed": config["seed"], "scale_steps": config["scale_steps"], "chunk_groups": config["chunk_groups"],
            "tile_elements": config["tile_elements"], "encoder": "torch", "device": config["device"],
            "compute_dtype": "float32", "bank_kind": "triality", "no_mtp": True,
            "max_materialized_bytes": config["max_materialized_bytes"],
            "application_code": {name: bootstrap.sha256(Path(__file__).with_name(name))
                                 for name in ("pipeline.py", "hf_stream.py")},
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "transformers", "huggingface-hub", "safetensors")},
            "runtime": runtime_identity(build)}


def check_conversion(config: dict, build: dict | None = None) -> dict:
    locations = paths(config)
    build = build if build is not None else build_gate(config)
    if not locations["output"].is_file() or not locations["conversion"].is_file():
        raise RuntimeError("A completed GGUF and its conversion report are required")
    report = json.loads(locations["conversion"].read_text())
    fingerprint = digest_json(conversion_identity(config, build))
    if (report.get("status") != "COMPLETE" or report.get("config_fingerprint") != fingerprint
            or report.get("schema") != SCHEMA):
        raise RuntimeError("Existing GGUF/report belongs to another configuration; choose a new output_name")
    artifact = report.get("output", {})
    output = locations["output"]
    if (output.is_symlink() or output.stat().st_size != artifact.get("bytes")
            or bootstrap.sha256(output) != artifact.get("sha256")):
        raise RuntimeError("GGUF changed after conversion; preserving it. Choose a new output_name")
    return report


def convert(config: dict) -> dict:
    build = build_gate(config)
    identity = conversion_identity(config, build)
    fingerprint = digest_json(identity)
    locations = paths(config)
    if locations["output"].exists() or locations["output"].is_symlink():
        # Reuse must not silently bless a same-sized change in an immutable
        # downloaded source. The converter performs this pass on new outputs.
        from colab_app.hf_stream import _snapshot_identity
        for spec, identity_source in zip(source_specs(config), identity["sources"]):
            _snapshot_identity(spec["directory"], identity_source["revision"])
        result = check_conversion(config, build)
        print("Reusing the verified matching GGUF: " + str(locations["output"]), flush=True)
        return result
    if locations["conversion"].exists():
        old = json.loads(locations["conversion"].read_text())
        if old.get("config_fingerprint") != fingerprint or old.get("status") == "COMPLETE":
            raise RuntimeError("An existing conversion report belongs to another run; choose a new output_name")
    sources = source_specs(config)
    cache = locations["work"] / "cache" / fingerprint
    locations["output"].parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": SCHEMA, "status": "RUNNING", "started_utc": timestamp(),
              "config_fingerprint": fingerprint, "identity": identity,
              "cache_dir": str(cache), "math_improvement_measured": False,
              "refusal_behavior_measured": False, "answer_data_used": False}
    bootstrap.atomic_json(locations["conversion"], report)
    try:
        from colab_app.hf_stream import convert_hf_to_spin8
        result = convert_hf_to_spin8(sources[0]["directory"], locations["output"],
                    llama_cpp_dir=locations["repo"], cache_dir=cache,
                    donor_dir=sources[1]["directory"] if len(sources) > 1 else None,
                    alpha=config["math_alpha"] if config["math_delta"] else 0.0,
                    direction_path=config["direction_path"] or None,
                    source_revision=identity["sources"][0]["revision"],
                    donor_revision=identity["sources"][1]["revision"] if len(sources) > 1 else None,
                    encoder="torch", device=config["device"], seed=config["seed"],
                    scale_steps=config["scale_steps"], chunk_groups=config["chunk_groups"],
                    tile_elements=config["tile_elements"], max_materialized_bytes=config["max_materialized_bytes"])
        if result.get("status") != "converted" or not locations["output"].is_file():
            raise RuntimeError("Streaming converter did not complete")
        output = bootstrap.file_info(locations["output"])
        if output["sha256"] != result.get("output_sha256"):
            raise RuntimeError("Output hash differs from the streaming converter result")
        report.update(status="COMPLETE", completed_utc=timestamp(), output=output, conversion=result)
    except BaseException as error:
        report.update(status="FAILED", error_type=type(error).__name__, failed_utc=timestamp())
        bootstrap.atomic_json(locations["conversion"], report)
        raise
    bootstrap.atomic_json(locations["conversion"], report)
    return report


def validate_native_report(metadata: dict, requested_device: str) -> dict:
    if metadata.get("status") != "PASS" or metadata.get("finite_logits") is not True:
        raise RuntimeError("Native model execution did not produce finite logits")
    shape = metadata.get("logits_shape")
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)):
        raise RuntimeError("Native logit shape is invalid")
    path = Path(metadata["logits_file"])
    count = math.prod(shape)
    if not path.is_file() or path.stat().st_size != count * 4:
        raise RuntimeError("Native logit byte size differs from its report")
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    dtype = metadata.get("logits_dtype")
    if dtype not in {"<f4", ">f4"}:
        raise RuntimeError("Native logit byte order is unspecified")
    if (dtype == "<f4") != (sys.byteorder == "little"):
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Native logit payload contains NaN or infinity")
    placements = [item for item in metadata.get("observed_output_buffer_placements", [])
                  if item.get("op") == "MUL_MAT_SPIN8" and item.get("completed_observations", 0) > 0]
    required = "cuda" if requested_device == "cuda" else "cpu"
    if not placements or any(required not in str(item.get("device", "")).lower() for item in placements):
        raise RuntimeError("Actual MUL_MAT_SPIN8 operation output placement does not match " + required)
    return {"finite_logits": True, "logit_elements": count, "required_operation": "MUL_MAT_SPIN8",
            "required_device": required, "actual_device_verified": True, "placements": placements,
            "logits": bootstrap.file_info(path), "quality_measured": False}


def verify(config: dict) -> dict:
    build = build_gate(config)
    conversion = check_conversion(config, build)
    locations = paths(config)
    report = {"schema": SCHEMA, "status": "RUNNING", "started_utc": timestamp(),
              "config_fingerprint": conversion["config_fingerprint"], "output": conversion["output"],
              "runtime": runtime_identity(build), "device": config["device"],
              "scope": "Converted model load, finite logits, actual native operation placement and API generation; no quality benchmark",
              "math_improvement_measured": False, "refusal_behavior_measured": False, "quality_measured": False}
    bootstrap.atomic_json(locations["verification"], report)
    directory = locations["reports"] / (locations["output"].stem + "-verify")
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "native"
    command = [build["test-spin8-native"]["path"], "--model", str(locations["output"]),
               "--output-prefix", str(prefix), "--ngl", "99" if config["device"] == "cuda" else "0",
               "--threads", "2", "--ctx", "128", "--flash-attn", "off",
               "--prompt-tokens", "1,4,7,9,13", "--decode-tokens", "17,21,25"]
    try:
        bootstrap.run(command, log=directory / "native-console.log", console_limit=25)
        metadata = json.loads(prefix.with_suffix(".json").read_text())
        report["native"] = validate_native_report(metadata, config["device"])
        report["native_metadata"] = bootstrap.file_info(prefix.with_suffix(".json"))
        report["api"] = bootstrap.server_smoke(Path(build["llama-server"]["path"]), locations["output"],
                                               directory / "server", config["device"], False)
        # The reusable bootstrap helper describes its own synthetic fixtures.
        # Here it was run on the actual converted model, with a smoke prompt.
        report["api"]["scope"] = "Converted model API generation smoke; not a math or refusal benchmark"
        bootstrap.atomic_json(directory / "server/server-test.json", report["api"])
        if report["api"]["status"] != "PASS":
            raise RuntimeError("Server generation APIs did not pass")
        report.update(status="PASS", completed_utc=timestamp())
    except BaseException as error:
        report.update(status="FAIL", failed_utc=timestamp(), error_type=type(error).__name__)
        bootstrap.atomic_json(locations["verification"], report)
        raise
    bootstrap.atomic_json(locations["verification"], report)
    return report


def publication_gate(config: dict) -> tuple[dict, dict, dict]:
    build = build_gate(config)
    conversion = check_conversion(config, build)
    path = paths(config)["verification"]
    if not path.is_file():
        raise RuntimeError("Run verify before publishing")
    verification = json.loads(path.read_text())
    if (verification.get("status") != "PASS" or verification.get("config_fingerprint") != conversion["config_fingerprint"]
            or verification.get("output", {}).get("sha256") != conversion["output"]["sha256"]
            or verification.get("runtime") != runtime_identity(build) or verification.get("device") != config["device"]
            or verification.get("native", {}).get("actual_device_verified") is not True
            or verification.get("native", {}).get("finite_logits") is not True
            or verification.get("api", {}).get("status") != "PASS"):
        raise RuntimeError("Verification does not match the current output/configuration/runtime")
    return build, conversion, verification


def _matching_benchmark(config: dict, conversion: dict, verification: dict, build: dict):
    """Accept optional measurements only for this verified GGUF and executable."""
    directory = paths(config)["work"] / "reports/benchmarks" / paths(config)["output"].stem
    report_path, markdown = directory / "benchmark.json", directory / "benchmark.md"
    if not report_path.exists():
        return None
    try:
        if not report_path.is_file() or report_path.is_symlink() or not markdown.is_file() or markdown.is_symlink():
            raise ValueError("benchmark JSON/Markdown is missing or not a regular file")
        report = json.loads(report_path.read_text())
        candidate = report.get("inputs", {}).get("candidate", {})
        server = report.get("inputs", {}).get("server", {})
        output_sha = conversion.get("output", {}).get("sha256")
        server_sha = build.get("llama-server", {}).get("sha256")
        if (report.get("schema") != "spin8-gsm8k-benchmark-v1" or report.get("status") != "COMPLETE"
                or candidate.get("sha256") != output_sha or server.get("sha256") != server_sha
                or verification.get("status") != "PASS" or verification.get("output", {}).get("sha256") != output_sha
                or verification.get("config_fingerprint") != conversion.get("config_fingerprint")
                or verification.get("runtime") != runtime_identity(build)):
            raise ValueError("benchmark is incomplete or does not match the current verified model/runtime")
        dataset = report.get("dataset", {})
        if (dataset.get("repo") != "openai/gsm8k"
                or not re.fullmatch(r"[0-9a-f]{40}", dataset.get("revision", ""))
                or not re.fullmatch(r"[0-9a-f]{64}", dataset.get("file_sha256", ""))):
            raise ValueError("benchmark dataset identity is not pinned")
        models = report.get("models", {})
        summary = models.get("candidate", {}).get("summary", {})
        requested = summary.get("requested_items")
        if (type(requested) is not int or requested <= 0 or summary.get("completed_items") != requested
                or type(summary.get("correct")) is not int or not 0 <= summary["correct"] <= requested
                or not isinstance(summary.get("accuracy"), (int, float))
                or not math.isfinite(summary["accuracy"])
                or not math.isclose(summary["accuracy"], summary["correct"] / requested, abs_tol=1e-12)):
            raise ValueError("benchmark candidate summary is incomplete/inconsistent")
        paired = report.get("paired", {})
        if paired.get("baseline_present"):
            baseline = models.get("baseline", {}).get("summary", {})
            if (baseline.get("requested_items") != requested or baseline.get("completed_items") != requested
                    or not re.fullmatch(r"[0-9a-f]{64}", report.get("inputs", {}).get("baseline", {}).get("sha256", ""))):
                raise ValueError("benchmark baseline comparison is incomplete")
        elif paired.get("comparison_measured") or paired.get("improvement_measured"):
            raise ValueError("candidate-only benchmark claims a before/after comparison")
        if report.get("artifacts", {}).get("benchmark_md_sha256") != bootstrap.sha256(markdown):
            raise ValueError("benchmark Markdown does not match the completed JSON report")
        return report, [("benchmarks/benchmark.json", report_path), ("benchmarks/benchmark.md", markdown)]
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("Optional benchmark evidence was not included: " + str(error), flush=True)
        return None


def matching_benchmark_files(config: dict, conversion: dict, verification: dict, build: dict) -> list[tuple[str, Path]]:
    """Shared HF/Drive allow-list; never includes the large floating baseline."""
    matched = _matching_benchmark(config, conversion, verification, build)
    return matched[1] if matched else []


def release_files(config: dict, build: dict, conversion: dict, verification: dict) -> list[tuple[str, Path]]:
    locations = paths(config)
    directory = locations["work"] / "release" / conversion["config_fingerprint"]
    directory.mkdir(parents=True, exist_ok=True)
    matched_benchmark = _matching_benchmark(config, conversion, verification, build)
    benchmark = matched_benchmark[0] if matched_benchmark else None
    benchmark_files = matched_benchmark[1] if matched_benchmark else []
    comparison_measured = bool(benchmark and benchmark.get("paired", {}).get("comparison_measured"))
    provenance = {"schema": SCHEMA, "created_utc": timestamp(), "output": conversion["output"],
                  "identity": conversion["identity"], "conversion": conversion["conversion"],
                  "verification": verification, "build": build,
                  "benchmark": {"report": "benchmarks/benchmark.json", "dataset": benchmark["dataset"],
                                "candidate": benchmark["models"]["candidate"]["summary"],
                                "paired": benchmark.get("paired", {}),
                                "artifacts": {name: bootstrap.file_info(path) for name, path in benchmark_files}}
                               if benchmark else None,
                  "limitations": {"standard_llama_cpp_compatible": False,
                      "math_comparison_measured": comparison_measured,
                      "refusal_behavior_measured": False, "answer_data_used": False,
                      "direction_semantics_verified": False, "quality_benchmark_run": bool(benchmark),
                      "generalized_math_ability_improvement_established": False}}
    bootstrap.atomic_json(directory / "provenance.json", provenance)
    payload = Path(config["payload_dir"])
    manifest = bootstrap.validate_payload(payload)
    if manifest["patch_sha256"] != build["patch_sha256"]:
        raise RuntimeError("Release payload patch differs from the verified runtime")
    source = conversion["identity"]["sources"][0]
    sources = conversion["identity"]["sources"]
    base_models = "\n".join("  - " + json.dumps(item["repo"]) for item in sources)
    direction = conversion["identity"]["direction_sha256"] is not None
    benchmark_section = "No matching completed quality benchmark is included. Mathematical ability has not been measured in this release."
    if benchmark:
        summary = benchmark["models"]["candidate"]["summary"]
        benchmark_section = (f"A matching GSM8K exact-answer benchmark completed {summary['completed_items']} items: "
                             f"{summary['correct']} correct ({100 * summary['accuracy']:.2f}%). "
                             f"A paired baseline comparison was measured: `{comparison_measured}`.\n\n"
                             "See [benchmark results](benchmarks/benchmark.md) and [complete measurements](benchmarks/benchmark.json) "
                             "for sample selection, prompt, decode budget, uncertainty and any baseline delta. "
                             "A completed comparison does not by itself establish an improvement, and the measurement "
                             "does not establish generalized mathematical ability.")
    card = f'''---
base_model:
{base_models}
library_name: gguf
tags:
  - gguf
  - experimental
  - spin8
  - direct-weight-edit
---
# {locations["output"].stem}

Experimental private Spin8 G128 GGUF produced directly from pinned HF Safetensors.
The payload uses 2.125 bits per quantized weight group, excluding rotation banks,
remaining floating tensors and metadata. The complete GGUF size is recorded in
`provenance.json`. It requires the supplied private llama.cpp runtime patch.
Standard llama.cpp, Ollama and other GGUF readers have not been validated for this ABI.

## Source and operations

- Source: [{source["repo"]}](https://huggingface.co/{source["repo"]}/tree/{source["revision"]})
- Source commit: `{source["revision"]}`
- Optional math interpolation enabled: `{config["math_delta"]}`
- Math alpha: `{conversion["identity"]["math_alpha"]}`
- Supplied residual direction projected out: `{direction}`
- Answer data used: `false`; optimizer steps: `0`.
- Codec bank: Spin(8) triality parameterization, seed `{config["seed"]}`.
- Full floating GGUF staged: `false`.

Math editing uses `W_source + alpha * (W_donor - W_source)` only with the audited
ancestor and donor pins. A supplied direction is removed from supported residual
writers and embeddings; the runtime projection metadata is also required.
The numerical vector alone does not establish that it represents refusal.

## Validation and limits

The native runtime passed its tiny codec, graph, and projected graph checks.
This output passed loading, finite logits, observed `MUL_MAT_SPIN8` output-buffer
placement on `{config["device"]}`, and both generation APIs. This is an execution
smoke test. Refusal behavior and long-context stability have not been benchmarked.
No generalized capability improvement is claimed. The placement callback is not
a GPU kernel profiler. Token rates in a benchmark, when available, are specific
to the recorded runtime and settings.

### Mathematical benchmark

{benchmark_section}

## Runtime

Upstream commit: `{bootstrap.BASE_COMMIT}`. Patch SHA-256: `{build["patch_sha256"]}`.
The `runtime/` folder contains the patch, manifest, fixtures and application code.
On a current Colab GPU runtime with its existing CUDA PyTorch installation:

```bash
python runtime/colab_app/bootstrap.py prepare --work-dir /content/spin8-runtime --payload-dir runtime --device cuda
/content/spin8-runtime/llama.cpp/build-spin8-v2-cuda/bin/llama-server -m {config["output_name"]} -ngl 99 -c 2048 --no-ui --flash-attn off
```

`--device cpu` builds and tests the CPU runtime. Native diagnostics and source,
toolchain, dependency, binary and model hashes are in `provenance.json`.
The server API is provided; an embedded web UI is disabled by default.

## Upstream licensing

Consult the pinned source model cards and supplied `upstream/` license files.
All original terms continue to apply to the source weights and their derivatives.
The native runtime's upstream MIT license is included in `runtime/licenses/LICENSE-llama.cpp`.
'''
    (directory / "README.md").write_text(card, encoding="utf-8")
    files = [(config["output_name"], locations["output"]), ("README.md", directory / "README.md"),
             ("provenance.json", directory / "provenance.json"),
             ("conversion.json", locations["conversion"]), ("verification.json", locations["verification"]),
             ("runtime/MANIFEST.json", payload / "MANIFEST.json")]
    files += benchmark_files
    entries = manifest.get("files", [])
    names = {item["path"] for item in entries}
    required = {"spin8-llama.patch", "licenses/LICENSE-llama.cpp",
                *("colab_app/" + name for name in ("bootstrap.py", "hf_fetch.py", "hf_stream.py", "pipeline.py")),
                *(item["path"] for item in manifest["fixture_files"])}
    if len(names) != len(entries) or not required <= names:
        raise RuntimeError("Release payload manifest is incomplete or duplicated; rebuild the notebook payload")
    for item in entries:
        file = bootstrap.safe_payload_file(payload, item["path"])
        if file.stat().st_size != item["bytes"] or bootstrap.sha256(file) != item["sha256"]:
            raise RuntimeError("Release payload file differs from its manifest: " + item["path"])
        files.append(("runtime/" + item["path"], file))
    for name in ("bootstrap.py", "hf_fetch.py", "hf_stream.py", "pipeline.py"):
        if bootstrap.sha256(payload / "colab_app" / name) != bootstrap.sha256(Path(__file__).with_name(name)):
            raise RuntimeError("Release payload app differs from executing code; rebuild the notebook payload: " + name)
    for spec in source_specs(config):
        for name in ("README.md", "LICENSE", "LICENSE.txt", "NOTICE"):
            path = spec["directory"] / name
            if path.is_file():
                files.append(("upstream/" + spec["role"] + "/" + name, path))
    return files


def publish(config: dict) -> dict:
    if not config["hf_repo"]:
        raise ValueError("Set hf_repo explicitly before publishing")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN in Colab Secrets/environment with write access; never put it in the JSON config")
    build, conversion, verification = publication_gate(config)
    files = release_files(config, build, conversion, verification)
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi(token=token)
    api.whoami()
    try:
        info = api.repo_info(config["hf_repo"], repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(config["hf_repo"], repo_type="model", private=config["hf_private"], exist_ok=False)
        info = api.repo_info(config["hf_repo"], repo_type="model")
    if bool(info.private) != config["hf_private"]:
        raise RuntimeError("Existing HF repository visibility differs from hf_private; choose the intended repository/configuration")
    commit = api.create_commit(repo_id=config["hf_repo"], repo_type="model",
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files],
                commit_message="Upload verified private Spin8 GGUF and reproducible runtime")
    report = {"schema": SCHEMA, "status": "PUBLISHED", "completed_utc": timestamp(),
              "repo": config["hf_repo"], "private": config["hf_private"], "commit": commit.oid,
              "url": str(commit.commit_url), "output_sha256": conversion["output"]["sha256"],
              "files": [name for name, _ in files], "token_stored": False}
    bootstrap.atomic_json(paths(config)["reports"] / "publication.json", report)
    return report


@contextmanager
def pipeline_lock(config: dict):
    import fcntl
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".spin8-pipeline.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline stage is running in this work directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("stage", choices=("plan", "download", "convert", "verify", "publish"))
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        with pipeline_lock(config):
            result = globals()[args.stage](config)
        print(json.dumps({"stage": args.stage, "status": result["status"],
                          "output": str(paths(config)["output"])}, indent=2), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Pipeline {args.stage} failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
