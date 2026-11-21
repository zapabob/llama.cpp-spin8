#!/usr/bin/env python3
"""Check native GGUF conversion against the independent reference representation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import convert as bridge
import gguf
import reference_codec as codec
import reference_gguf


def compare_tensors(left: gguf.GGUFReader, right: gguf.GGUFReader) -> None:
    left_map = {tensor.name: tensor for tensor in left.tensors}
    right_map = {tensor.name: tensor for tensor in right.tensors}
    assert left_map.keys() == right_map.keys()
    for name, a in left_map.items():
        b = right_map[name]
        assert a.tensor_type == b.tensor_type, name
        assert np.array_equal(a.shape, b.shape), name
        assert a.data.tobytes() == b.data.tobytes(), name


def run_checks(directory: Path) -> dict:
    source = directory / "tiny-qwen35-f32.gguf"
    native = directory / "tiny-qwen35-spin8.gguf"
    control = directory / "tiny-qwen35-decoded.gguf"
    reference = directory / "tiny-qwen35-reference.gguf"
    source_before = bridge.sha256(source)
    original, packed, decoded = (gguf.GGUFReader(path) for path in (source, native, control))
    originals = {tensor.name: tensor for tensor in original.tensors}
    packed_map = {tensor.name: tensor for tensor in packed.tensors}
    decoded_map = {tensor.name: tensor for tensor in decoded.tensors}
    checks = []
    for name, field in original.fields.items():
        if name.startswith(("qwen35.", "tokenizer.")):
            assert packed.fields[name].types == field.types
            assert packed.fields[name].contents() == field.contents()
    checks.append("canonical architecture and tokenizer metadata preserved with exact types")
    assert packed.fields["spin8.codec_version"].types == [gguf.GGUFValueType.UINT32]
    assert packed.fields["spin8.codec_version"].contents() == 1
    assert packed.fields["general.architecture"].contents() == "qwen35"
    checks.append("native ABI metadata is explicit")
    reconstructed = 0
    for tensor in packed.tensors:
        if tensor.tensor_type == bridge.TYPE:
            bank = packed_map[tensor.name + bridge.BANK_SUFFIX]
            view = bridge.packed_view(tensor.data, bank.data, bridge.logical_shape(tensor))
            assert np.array_equal(tensor.shape, originals[tensor.name].shape)
            assert np.array_equal(codec.decode(view), decoded_map[tensor.name].data)
            reconstructed += 1
        elif not tensor.name.endswith(bridge.BANK_SUFFIX):
            assert tensor.data.tobytes() == originals[tensor.name].data.tobytes()
    assert reconstructed > 0
    checks.append("all native matrices preserve logical shapes and match decoded F32 weights")
    checks.append("unquantized tensor bytes preserved")
    overlay, overlay_metadata = reference_gguf.read_gguf(reference)
    assert overlay_metadata["project.canonical_gguf_sha256"] == source_before
    checks.append("reference overlay is bound to canonical source hash")
    with tempfile.TemporaryDirectory(prefix="spin8-conversion-test-") as temporary:
        work = Path(temporary)
        args = argparse.Namespace(source=native, output=work / "decoded.gguf", force=False, temp_dir=None)
        bridge.decode_model(args)
        compare_tensors(decoded, gguf.GGUFReader(args.output))
        checks.append("decode command reproduces all control tensor bytes")
        args = argparse.Namespace(command="transcode", source=source, reference=reference,
                                  output=work / "transcoded.gguf", decoded_control=None,
                                  force=False, temp_dir=None)
        bridge.convert(args)
        compare_tensors(packed, gguf.GGUFReader(args.output))
        checks.append("reference transcode preserves every native tensor byte")
        unmarked = work / "unmarked-reference.gguf"
        name = next(iter(overlay))
        reference_gguf.write_gguf(unmarked, {name: overlay[name]})
        args.reference, args.output = unmarked, work / "must-not-exist.gguf"
        try:
            bridge.convert(args)
        except ValueError:
            assert not args.output.exists()
        else:
            raise AssertionError("An unmarked HF-layout reference was accepted")
        checks.append("unmarked or HF-layout references rejected before output")

        selected = "blk.0.ffn_down.weight"
        bf16_source = work / "bf16-source.gguf"
        descriptors = []
        expected_values = None
        for tensor in original.tensors:
            if tensor.name == selected:
                array = gguf.quantize(tensor.data, gguf.GGMLQuantizationType.BF16)
                expected_values = gguf.dequantize(array, gguf.GGMLQuantizationType.BF16)
                descriptors.append((tensor.name, array, gguf.GGMLQuantizationType.BF16))
            else:
                descriptors.append((tensor.name, tensor.data, tensor.tensor_type))
        bridge.write_model(bf16_source, original, descriptors, {"test": "BF16 input"}, native=False, overwrite=False)
        args = argparse.Namespace(command="quantize", source=bf16_source, output=work / "bf16-native.gguf",
                                  decoded_control=work / "bf16-control.gguf", reference_output=None,
                                  include=[r"^blk\.0\.ffn_down\.weight$"], exclude=[], include_embedding=False,
                                  bank_kind="triality", seed=81237, scale_steps=3, chunk_groups=32,
                                  force=False, temp_dir=None)
        bridge.convert(args)
        bf16_native = gguf.GGUFReader(args.output)
        bf16_map = {tensor.name: tensor for tensor in bf16_native.tensors}
        actual = bridge.packed_view(bf16_map[selected].data, bf16_map[selected + bridge.BANK_SUFFIX].data,
                                    bridge.logical_shape(bf16_map[selected]))
        expected = codec.encode(expected_values, codec.make_bank("triality", 81237), scale_steps=3, chunk_groups=32)
        assert np.array_equal(actual["words"], expected["words"])
        assert np.array_equal(actual["scales"].view("<u2"), expected["scales"].view("<u2"))
        checks.append("BF16 source decoding and quantization agree with official gguf BF16 conversion")
        assert bridge.sha256(source) == source_before
        checks.append("source GGUF remains unchanged")
    report = {"status": "passed", "checks": checks, "checks_passed": len(checks),
              "native_tensor_count_checked": reconstructed, "runtime_inference_test": False,
              "scope": "Python export and payload parity", "source_sha256": source_before}
    (directory / "conversion-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.fixture_dir), indent=2))
