#!/usr/bin/env python3
"""Exercise cache reuse, corruption recovery and immutable run settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

import convert as bridge
import gguf
from resume import TensorCache


def run_checks(source: Path) -> dict:
    source_hash = bridge.sha256(source)
    checks = []
    with tempfile.TemporaryDirectory(prefix="spin8-resume-test-") as directory:
        work = Path(directory)
        args = argparse.Namespace(command="quantize", source=source, output=work / "first.gguf",
            decoded_control=None, reference_output=None, force=False, temp_dir=work,
            resume_dir=work / "cache", include=[r"^blk\.0\.ffn_(gate|up)\.weight$"], exclude=[],
            include_embedding=False, bank_kind="triality", seed=81237, scale_steps=3,
            chunk_groups=64, encoder="numpy", device="cpu", compute_dtype="float32")
        first = bridge.convert(args)
        assert first["resume"]["written"] == 2 and first["resume"]["hits"] == 0
        checks.append("completed tensors are atomically stored and validated")
        args.output = work / "second.gguf"
        second = bridge.convert(args)
        assert second["resume"]["hits"] == 2 and second["resume"]["written"] == 0
        assert first["output_sha256"] == second["output_sha256"]
        checks.append("unchanged rerun reuses both tensors and produces identical GGUF bytes")

        manifest = json.loads((args.resume_dir / "run.json").read_text())
        cache = TensorCache(args.resume_dir, manifest["settings"])
        name = "blk.0.ffn_gate.weight"
        payload, bank, record = cache._paths(name)
        unrelated = args.resume_dir / "keep-user-note.txt"
        unrelated.write_text("preserve\n")
        with payload.open("r+b") as target:
            first_byte = target.read(1)
            target.seek(0)
            target.write(bytes([first_byte[0] ^ 1]))
        args.output = work / "repaired.gguf"
        repaired = bridge.convert(args)
        assert repaired["resume"]["invalidated"] == 1 and repaired["resume"]["written"] == 1
        assert repaired["resume"]["hits"] == 1 and repaired["output_sha256"] == first["output_sha256"]
        assert unrelated.read_text() == "preserve\n"
        checks.append("hash-corrupt tensor is retried while the other tensor and unrelated files remain")

        # A matching file hash alone is insufficient: zero FP16 scale is illegal.
        with payload.open("r+b") as target:
            target.seek(32)
            target.write(b"\x00\x00")
        record_data = json.loads(record.read_text())
        record_data["payload_sha256"] = bridge.sha256(payload)
        record.write_text(json.dumps(record_data))
        args.output = work / "content-repaired.gguf"
        repaired = bridge.convert(args)
        assert repaired["resume"]["invalidated"] == 1 and repaired["resume"]["hits"] == 1
        assert repaired["output_sha256"] == first["output_sha256"]
        checks.append("packed content is validated in addition to sizes and SHA-256")

        record.unlink()
        partial = payload.with_name(payload.name + ".partial")
        partial.write_bytes(b"interrupted write")
        args.output = work / "partial-repaired.gguf"
        repaired = bridge.convert(args)
        assert repaired["resume"]["invalidated"] == 1 and repaired["resume"]["hits"] == 1
        assert not partial.exists() and repaired["output_sha256"] == first["output_sha256"]
        checks.append("a missing commit record and interrupted write retry only their tensor")

        manifest_bytes = (args.resume_dir / "run.json").read_bytes()
        args.seed += 1
        args.output = work / "must-not-exist.gguf"
        try:
            bridge.convert(args)
        except ValueError as error:
            assert "mismatch" in str(error)
            assert not args.output.exists()
            assert (args.resume_dir / "run.json").read_bytes() == manifest_bytes
        else:
            raise AssertionError("Changed settings reused an incompatible cache")
        checks.append("changed run settings fail before output and preserve the existing manifest")

        names = gguf.TensorNameMap(gguf.MODEL_ARCH.QWEN35, 5)
        fused = {"blk.0.attn_qkv_gate.weight": gguf.MODEL_TENSOR.ATTN_QKV_GATE,
                 "blk.2.ssm_beta_alpha.weight": gguf.MODEL_TENSOR.SSM_BETA_ALPHA_CAT}
        for name, kind in fused.items():
            tensor = SimpleNamespace(name=name, shape=np.array([128, 4]), tensor_type=gguf.GGMLQuantizationType.F32)
            assert bridge.projection_kind(tensor, names) == kind, name
            for shape, tensor_type in (([128], gguf.GGMLQuantizationType.F32),
                                       ([128, 4, 2], gguf.GGMLQuantizationType.F32),
                                       ([127, 4], gguf.GGMLQuantizationType.F32),
                                       ([128, 0], gguf.GGMLQuantizationType.F32),
                                       ([128, 4], gguf.GGMLQuantizationType.I32)):
                tensor.shape, tensor.tensor_type = np.array(shape), tensor_type
                assert bridge.projection_kind(tensor, names) is None, (name, shape, tensor_type)
        checks.append("canonical fused QKV/gate and beta/alpha select only eligible float matrices")
        for name in ("blk.0.attn_norm.weight", "blk.0.ssm_conv1d.weight", "unknown.weight"):
            tensor = SimpleNamespace(name=name, shape=np.array([128, 4]), tensor_type=gguf.GGMLQuantizationType.F32)
            assert bridge.projection_kind(tensor, names) is None
        checks.append("128-aligned normalization, convolution and unknown matrices are not selected")
        for name in ("blk.0.attn_qkv_gate.bias", "blk.0.ssm_beta_alpha.weight.extra",
                     "blk.0.ssm_beta_alpha_extra.weight", "blk.-1.attn_qkv_gate.weight",
                     "blk.00.attn_qkv_gate.weight", "prefix.blk.0.attn_qkv_gate.weight",
                     "model.layers.0.self_attn.q_proj.weight"):
            tensor = SimpleNamespace(name=name, shape=np.array([128, 4]), tensor_type=gguf.GGMLQuantizationType.F32)
            assert bridge.projection_kind(tensor, names) is None, name
        checks.append("noncanonical aliases, fused lookalikes and biases remain excluded")
    assert bridge.sha256(source) == source_hash
    return {"status": "PASS", "checks": checks, "source_sha256": source_hash,
            "check_count": len(checks), "cuda_executed": False, "quality_measured": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_checks(args.fixture)
    text = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
