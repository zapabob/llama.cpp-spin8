#!/usr/bin/env python3
"""Check graph metadata, auxiliary tensors and portable publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

import convert as bridge
import gguf
import make_fixture
from atomic_output import publish_file


def run_checks() -> dict:
    checks = []
    with tempfile.TemporaryDirectory(prefix="spin8-metadata-test-") as directory:
        work = Path(directory)
        source, output, control = (work / name for name in ("source.gguf", "native.gguf", "control.gguf"))
        tensors = make_fixture.fixture_weights()
        direction = np.zeros(128, dtype=np.float32)
        direction[0] = 1
        tensors["spin8.projection.direction"] = direction
        selected = "blk.0.ffn_down.weight"
        tensors[selected] = bridge.codec.hadamard128(tensors[selected].reshape(-1, 128)).reshape(tensors[selected].shape).astype(np.float32)
        writer = gguf.GGUFWriter(source, "qwen35")
        make_fixture.fixture_metadata(writer, "gpt2-byte")
        writer.add_uint32("prism.hadamard.version", 1)
        writer.add_uint32("prism.hadamard.block_size", 128)
        writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        writer.add_string("prism.hadamard.axis", "input-last-dimension")
        writer.add_string("prism.hadamard.sign_mode", "identity")
        writer.add_array("prism.hadamard.weight_names", [selected])
        writer.add_uint32("spin8.projection.version", 1)
        writer.add_array("spin8.projection.left_weights", [selected])
        writer.add_key_value("spin8.projection.right_weights", [], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
        writer.add_string("spin8.edit.recipe", "metadata preservation test, not a behavior experiment")
        for name, tensor in tensors.items():
            writer.add_tensor(name, tensor)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        args = argparse.Namespace(command="quantize", source=source, output=output,
            decoded_control=control, reference_output=None, force=False, temp_dir=work,
            include=["^blk\\.0\\.ffn_down\\.weight$"], exclude=[], include_embedding=False,
            bank_kind="triality", seed=81237, scale_steps=3, chunk_groups=64)
        bridge.convert(args)
        original = gguf.GGUFReader(source)
        for path in (output, control):
            result = gguf.GGUFReader(path)
            for name, field in original.fields.items():
                if name.startswith(("prism.hadamard.", "spin8.projection.", "spin8.edit.")):
                    assert result.fields[name].types == field.types, name
                    assert result.fields[name].contents() == field.contents(), name
            auxiliary = next(t for t in result.tensors if t.name == "spin8.projection.direction")
            assert auxiliary.tensor_type == gguf.GGMLQuantizationType.F32
            assert auxiliary.data.tobytes() == direction.tobytes()
        checks.append("native and decoded outputs preserve Prism/projection/edit metadata, including typed empty arrays")
        checks.append("the F32 auxiliary projection direction remains byte-identical")

        temporary, target = work / "complete.tmp", work / "published.bin"
        temporary.write_bytes(b"complete output")
        publish_file(temporary, target)
        assert target.read_bytes() == b"complete output" and not temporary.exists()
        temporary.write_bytes(b"replacement")
        try:
            publish_file(temporary, target)
        except FileExistsError:
            assert target.read_bytes() == b"complete output" and temporary.exists()
        else:
            raise AssertionError("No-overwrite publication replaced a file")
        lock = target.with_name(target.name + ".publish.lock")
        lock.write_text("another writer")
        try:
            publish_file(temporary, target, overwrite=True)
        except FileExistsError:
            assert target.read_bytes() == b"complete output" and lock.read_text() == "another writer"
        else:
            raise AssertionError("A publication lock was ignored")
        lock.unlink()
        publish_file(temporary, target, overwrite=True)
        assert target.read_bytes() == b"replacement" and not lock.exists()
        checks.append("portable publication enforces no-overwrite, exclusive lock and explicit replacement")
    return {"status": "PASS", "checks": checks, "runtime_inference_test": False, "drivefs_mounted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    text = json.dumps(run_checks(), indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
