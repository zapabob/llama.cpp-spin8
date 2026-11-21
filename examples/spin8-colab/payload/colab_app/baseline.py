#!/usr/bin/env python3
"""Create a reusable original-source F16 GGUF for paired accuracy evaluation.

This file is an optional benchmark reference, never an intermediate required by
the direct Spin8 converter. The source is the same pinned original checkpoint
used by the candidate, before math interpolation and direction projection.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colab_app import bootstrap, pipeline
from colab_app.hf_stream import _snapshot_identity

SCHEMA = "spin8-original-f16-baseline-v1"


def _float_inventory(path: Path, repo: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(path)
    architecture = reader.get_field("general.architecture")
    if architecture is None or architecture.contents() != "qwen35":
        raise ValueError("Benchmark reference is not a qwen35 GGUF")
    if not reader.tensors or any(tensor.tensor_type not in {
            gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32} for tensor in reader.tensors):
        raise ValueError("Benchmark reference must contain only canonical F16/F32 tensors")
    if any(key.startswith("spin8.") for key in reader.fields):
        raise ValueError("Benchmark reference unexpectedly contains Spin8 editing/codec metadata")
    return {"architecture": "qwen35", "tensor_count": len(reader.tensors),
            "tensor_types": sorted({tensor.tensor_type.name for tensor in reader.tensors}),
            "original_weights": True, "math_interpolation": False, "direction_projection": False,
            "mtp_included": False, "quality_measured": False}


def prepare_baseline(config: dict, *, skip_if_insufficient_space: bool = False) -> dict[str, Any]:
    build, conversion, _ = pipeline.publication_gate(config)
    locations = pipeline.paths(config)
    spec = pipeline.source_specs(config)[0]
    source_report = pipeline.read_source(spec)
    source = spec["directory"]
    repo = locations["repo"]
    converter = repo / "convert_hf_to_gguf.py"
    code_paths = [converter, repo / "conversion/base.py", repo / "conversion/qwen.py",
                  repo / "gguf-py/gguf/gguf_writer.py", repo / "gguf-py/gguf/vocab.py"]
    identity = {"schema": SCHEMA, "source_repo": source_report["repo"],
                "source_revision": source_report["revision"],
                "source_content_fingerprint": source_report["content_fingerprint"],
                "base_commit": build["base_commit"], "outtype": "f16", "no_mtp": True,
                "unmodified_source": True,
                "converter_code": {str(path.relative_to(repo)): bootstrap.sha256(path) for path in code_paths},
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "numpy", "transformers", "tokenizers", "safetensors")}}
    fingerprint = pipeline.digest_json(identity)
    directory = locations["work"] / "baselines"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("original-" + source_report["revision"][:12] + "-" + fingerprint[:12] + "-F16.gguf")
    manifest_path = output.with_suffix(".baseline.json")
    if output.exists() or output.is_symlink():
        if not manifest_path.is_file() or output.is_symlink():
            raise RuntimeError("Preserving an unverified existing F16 reference: " + str(output))
        old = json.loads(manifest_path.read_text())
        if (old.get("status") != "COMPLETE" or old.get("fingerprint") != fingerprint
                or old.get("schema") != SCHEMA):
            raise RuntimeError("Existing baseline manifest is incomplete or belongs to another source")
        artifact = bootstrap.file_info(output)
        if artifact["sha256"] != old.get("output", {}).get("sha256") or artifact["bytes"] != old["output"].get("bytes"):
            raise RuntimeError("Existing baseline file changed after preparation")
        old["reused"] = True
        print("Reusing original F16 benchmark reference: " + str(output), flush=True)
        return old
    # The official BF16/F16 source payload is a conservative size estimate for
    # the text-only F16 export. Add room for metadata and the user's disk reserve.
    estimated_file_bytes = max(source_report.get("verified_payload_bytes", 0),
                               source_report["planned_weight_file_bytes"]) + pipeline.GIB
    reserve = int(config["disk_reserve_gib"] * pipeline.GIB)
    free = shutil.disk_usage(directory).free
    report = {"schema": SCHEMA, "status": "RUNNING", "identity": identity,
              "fingerprint": fingerprint, "started_utc": pipeline.timestamp(),
              "candidate_sha256": conversion["output"]["sha256"],
              "estimated_file_bytes": estimated_file_bytes, "free_bytes_before": free,
              "reserve_bytes": reserve, "reused": False}
    if free < estimated_file_bytes + reserve:
        report.update(status="SKIPPED_SPACE", completed_utc=pipeline.timestamp(),
                      reason="Insufficient estimated disk for optional F16 reference; no paired comparison prepared")
        if skip_if_insufficient_space:
            return report
        raise RuntimeError(f"Optional F16 reference needs approximately {(estimated_file_bytes + reserve)/pipeline.GIB:.1f} GiB free; "
                           f"available {free/pipeline.GIB:.1f} GiB")
    bootstrap.atomic_json(manifest_path, report)
    fd, name = tempfile.mkstemp(prefix="." + output.stem + ".", suffix=".partial.gguf", dir=directory)
    os.close(fd)
    temporary = Path(name)
    try:
        # Re-check the bytes once before forming an accuracy reference. A
        # COMPLETE manifest alone does not prove unchanged local shard contents.
        source_identity = _snapshot_identity(source, source_report["revision"])
        if source_identity["content_fingerprint"] != source_report["content_fingerprint"]:
            raise RuntimeError("Baseline source bytes differ from the verified source snapshot")
        command = [sys.executable, "-u", str(converter), str(source), "--outfile", str(temporary),
                   "--outtype", "f16", "--no-nextn"]
        bootstrap.run(command, log=output.with_suffix(".conversion.log"), console_limit=30)
        report["canonical_inventory"] = _float_inventory(temporary, repo)
        artifact = bootstrap.file_info(temporary)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Another process created the F16 reference: " + str(output))
        with temporary.open("rb") as complete:
            os.fsync(complete.fileno())
        os.replace(temporary, output)
        artifact["path"] = str(output.resolve())
        report.update(status="COMPLETE", completed_utc=pipeline.timestamp(), output=artifact)
        bootstrap.atomic_json(manifest_path, report)
        print("Prepared original F16 benchmark reference: " + str(output), flush=True)
        return report
    except BaseException as error:
        report.update(status="FAILED", failed_utc=pipeline.timestamp(), error_type=type(error).__name__)
        bootstrap.atomic_json(manifest_path, report)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-if-insufficient-space", action="store_true")
    args = parser.parse_args()
    try:
        config = pipeline.load_config(args.config)
        with pipeline.pipeline_lock(config):
            report = prepare_baseline(config, skip_if_insufficient_space=args.skip_if_insufficient_space)
        bootstrap.atomic_json(args.report, report)
        print(json.dumps({"status": report["status"], "output": report.get("output", {}).get("path"),
                          "report": str(args.report)}, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        bootstrap.atomic_json(args.report, {"schema": SCHEMA, "status": "FAILED",
                                          "error_type": type(error).__name__, "failed_utc": pipeline.timestamp()})
        print(f"F16 baseline preparation failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
