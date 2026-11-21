#!/usr/bin/env python3
"""Numerical and input-contract checks for direct weight edits on synthetic weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from convert import gguf, logical_shape, sha256
from edit_weights import edit, DIRECTION_NAME, projection_axis
from make_fixture import fixture_weights, write_fixture


def command(source, output, **kwargs):
    settings = dict(source=source, output=output, math_base=None, math_donor=None, alpha=0.5,
                    math_include=[], preserve_missing=[], remove_direction=None,
                    outtype="f32", tile_elements=512, temp_dir=output.parent, force=False)
    settings.update(kwargs)
    return argparse.Namespace(**settings)


def tensors(path):
    reader = gguf.GGUFReader(path)
    return reader, {tensor.name: tensor.data.reshape(logical_shape(tensor)).copy() for tensor in reader.tensors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix="spin8-direct-edit-check-") as directory:
        work = Path(directory)
        rng = np.random.default_rng(20261120)
        base_weights = fixture_weights()
        fused = {"blk.0.attn_qkv_gate.weight": ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"),
                 "blk.0.ssm_beta_alpha.weight": ("blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight")}
        for name, parts in fused.items():
            base_weights[name] = np.concatenate([base_weights.pop(part) for part in parts], axis=0)
        donor_weights = {name: array.copy() for name, array in base_weights.items()}
        for name, array in donor_weights.items():
            if array.ndim == 2 and name.endswith(".weight"):
                donor_weights[name] = (array + rng.normal(0, 0.001, array.shape)).astype(np.float32)
        base, donor = work / "base.gguf", work / "donor.gguf"
        write_fixture(base, base_weights)
        write_fixture(donor, donor_weights)
        original_hash = sha256(base)
        direction = rng.normal(size=128)
        direction_file = work / "direction.npy"
        np.save(direction_file, direction)
        identity = work / "identity.gguf"
        edit(command(base, identity, math_base=base, math_donor=donor, alpha=0))
        _, identity_values = tensors(identity)
        for name, value in base_weights.items():
            np.testing.assert_array_equal(identity_values[name], value)
        checks.append("alpha_zero_preserves_all_numeric_tensors")
        output = work / "combined.gguf"
        report = edit(command(base, output, math_base=base, math_donor=donor, alpha=0.25,
                              remove_direction=direction_file))
        reader, result = tensors(output)
        d = result[DIRECTION_NAME].astype(np.float64)
        projector = np.eye(128) - np.outer(d, d) / (d @ d)
        records = {entry["name"]: entry for entry in report["recipe"]["edited_tensors"]}
        for name in fused:
            assert name in records and records[name]["task_delta_applied"], name
        checks.append("canonical fused QKV/gate and beta/alpha receive the requested task delta")
        max_abs = 0.0
        for name, original in base_weights.items():
            expected = original.astype(np.float64)
            if name in records and records[name]["task_delta_applied"]:
                expected += 0.25 * (donor_weights[name].astype(np.float64) - original)
            if projection_axis(name) == "left":
                expected = projector @ expected
            elif projection_axis(name) == "right":
                expected = expected @ projector
            np.testing.assert_allclose(result[name], expected, rtol=2e-7, atol=2e-8)
            max_abs = max(max_abs, float(np.max(np.abs(result[name] - expected))))
        checks.append("direct_task_delta_and_both_projection_axes_match_explicit_dense_projector")
        assert reader.get_field("spin8.projection.version").contents() == 1
        assert reader.get_field("spin8.projection.right_weights").contents() == ["token_embd.weight"]
        assert len(reader.get_field("spin8.projection.left_weights").contents()) == 8
        checks.append("projection_metadata_matches_edited_weights")
        assert report["recipe"]["optimizer_steps"] == 0 and report["recipe"]["answer_data_used"] is False
        assert report["math_improvement_measured"] is False and report["refusal_behavior_measured"] is False
        checks.append("report_does_not_claim_semantic_improvement_or_direction_identification")
        output16 = work / "projection-f16.gguf"
        edited16 = edit(command(base, output16, remove_direction=direction_file, outtype="f16"))
        assert all(np.isfinite(record["stored_projection_energy_fraction"]) for record in edited16["recipe"]["edited_tensors"])
        assert max(record["stored_projection_energy_fraction"] for record in edited16["recipe"]["edited_tensors"]) > 0
        checks.append("f16_rounding_leakage_is_measured_and_runtime_projection_required")
        missing_weights = {name: array for name, array in donor_weights.items() if name != "blk.0.ffn_up.weight"}
        missing = work / "missing.gguf"
        write_fixture(missing, missing_weights)
        invalid_cases = [
            ("missing_donor_tensor", command(base, work / "bad-missing.gguf", math_base=base, math_donor=missing)),
            ("nonfinite_alpha", command(base, work / "bad-alpha.gguf", math_base=base, math_donor=donor, alpha=float("nan"))),
            ("source_overwrite", command(base, base, remove_direction=direction_file, force=True)),
            ("preexisting_projection", command(output, work / "bad-double.gguf", remove_direction=direction_file)),
        ]
        for name, value in (("zero", np.zeros(128)), ("wrong_width", np.ones(127)), ("nan", np.full(128, np.nan))):
            path = work / (name + ".npy")
            np.save(path, value)
            invalid_cases.append((name + "_direction", command(base, work / ("bad-" + name + ".gguf"), remove_direction=path)))
        for name, settings in invalid_cases:
            try:
                edit(settings)
            except (ValueError, FileExistsError):
                checks.append("reject_" + name)
            else:
                raise AssertionError("accepted " + name)
        preserved = work / "preserved.gguf"
        preserve_report = edit(command(base, preserved, math_base=base, math_donor=missing,
                                       preserve_missing=[r"blk\.0\.ffn_up\.weight"]))
        _, preserved_values = tensors(preserved)
        np.testing.assert_array_equal(preserved_values["blk.0.ffn_up.weight"], base_weights["blk.0.ffn_up.weight"])
        assert preserve_report["recipe"]["preserved_missing_tensors"] == ["blk.0.ffn_up.weight"]
        checks.append("explicit_missing_tensor_preservation_is_recorded")
        assert sha256(base) == original_hash
        checks.append("original_source_is_unchanged")
        result_report = {"status": "PASS", "checks": checks, "count": len(checks),
                         "explicit_projector_max_abs_error": max_abs,
                         "scope": "Synthetic direct-weight arithmetic and projection; no real math or refusal behavior measured"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result_report, indent=2) + "\n")
    print(json.dumps(result_report))


if __name__ == "__main__":
    main()
