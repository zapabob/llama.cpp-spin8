#!/usr/bin/env python3
"""CPU integration parity: real HF Safetensors → official Qwen transforms → Spin8.

The tiny generated tokenizer is explicitly labelled gpt-2 only in this test.
That override is shared by both stock and streaming paths; no model or numeric
conversion is mocked. These tests do not establish pretrained model quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import hf_stream


def make_hf(directory: Path, modules: dict, *, donor=False, dtype="bfloat16") -> None:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    directory.mkdir(parents=True)
    rng = np.random.default_rng(91261)
    tensors = {}

    def add(name, shape, *, norm=False, scale=0.04):
        array = rng.normal(0, scale, shape).astype(np.float32)
        if norm:
            array *= 0.1
        if donor and len(shape) == 2:
            array += 0.003 * np.sin(np.arange(array.size).reshape(shape)).astype(np.float32)
        tensors[name] = torch.from_numpy(array).to(getattr(torch, dtype))

    add("model.language_model.embed_tokens.weight", (256, 128), scale=0.1)
    add("model.language_model.norm.weight", (128,), norm=True)
    add("lm_head.weight", (256, 128), scale=0.06)
    for layer in range(2):
        prefix = f"model.language_model.layers.{layer}."
        add(prefix + "input_layernorm.weight", (128,), norm=True)
        add(prefix + "post_attention_layernorm.weight", (128,), norm=True)
        add(prefix + "mlp.gate_proj.weight", (256, 128))
        add(prefix + "mlp.up_proj.weight", (256, 128))
        add(prefix + "mlp.down_proj.weight", (128, 256))
        if layer == 0:
            for suffix, shape in {
                "in_proj_qkv.weight": (256, 128), "in_proj_z.weight": (128, 128),
                "in_proj_a.weight": (4, 128), "in_proj_b.weight": (4, 128),
                "out_proj.weight": (128, 128), "conv1d.weight": (256, 1, 4),
                "dt_bias": (4,), "A_log": (4,), "norm.weight": (32,),
            }.items():
                add(prefix + "linear_attn." + suffix, shape)
        else:
            for suffix, shape in {
                "q_proj.weight": (256, 128), "k_proj.weight": (64, 128),
                "v_proj.weight": (64, 128), "o_proj.weight": (128, 128),
                "q_norm.weight": (64,), "k_norm.weight": (64,),
            }.items():
                add(prefix + "self_attn." + suffix, shape, norm="norm" in suffix)
    # Vision weights must be ignored by the official text converter.
    add("model.visual.patch_embed.weight", (8, 8))
    save_file(tensors, directory / "model.safetensors")
    text_config = {
        "model_type": "qwen3_5_text", "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 256,
        "max_position_embeddings": 128, "tie_word_embeddings": False,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
        "linear_value_head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"], "mtp_num_hidden_layers": 0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 1.0, "mrope_section": [8, 8, 16]},
    }
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": text_config, "tie_word_embeddings": False}
    (directory / "config.json").write_text(json.dumps(config))
    mapping = modules["gguf"].vocab.bytes_to_unicode()
    tokenizer = Tokenizer(models.BPE(vocab={mapping[i]: i for i in range(256)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=mapping[0], eos_token=mapping[0],
                                  unk_token=mapping[0], pad_token=mapping[0])
    fast.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
    fast.save_pretrained(directory)


@contextmanager
def synthetic_tokenizer(modules):
    with patch.object(modules["conversion"].TextModel, "get_vocab_base_pre", return_value="gpt-2"):
        yield


def stock_float(directory, output, modules):
    original = modules["conversion"].get_model_class("Qwen3_5ForConditionalGeneration")
    model_class = type("StockTestQwen", (original,), {"model_arch": original.model_arch, "no_mtp": True})
    model = model_class(directory, modules["gguf"].LlamaFileType.MOSTLY_F16, output, eager=False)
    model.write()


def stock_quantize(source, output, modules, **kwargs):
    config = dict(command="quantize", source=source, output=output, force=False, temp_dir=None,
                  decoded_control=None, reference_output=None, resume_dir=None, include=[], exclude=[],
                  include_embedding=True, bank_kind="triality", seed=81237, scale_steps=3,
                  chunk_groups=16, encoder="numpy", device="cpu", compute_dtype="float64")
    config.update(kwargs)
    return modules["convert"].convert(argparse.Namespace(**config))


def compare(left, right, modules):
    a, b = (modules["gguf"].GGUFReader(path) for path in (left, right))
    a_map, b_map = ({tensor.name: tensor for tensor in reader.tensors} for reader in (a, b))
    assert a_map.keys() == b_map.keys(), (a_map.keys() - b_map.keys(), b_map.keys() - a_map.keys())
    for name, x in a_map.items():
        y = b_map[name]
        assert x.tensor_type == y.tensor_type, name
        np.testing.assert_array_equal(x.shape, y.shape, err_msg=name)
        np.testing.assert_array_equal(x.data, y.data, err_msg=name)
    for key in a.fields:
        if key.startswith(("qwen35.", "tokenizer.", "spin8.projection.")):
            assert a.fields[key].types == b.fields[key].types, key
            assert a.fields[key].contents() == b.fields[key].contents(), key
    return len(a_map)


def checks(root: Path, work: Path, *, encoder="numpy", device="cpu") -> dict:
    modules = hf_stream._load_modules(root)
    source, donor = work / "hf-base", work / "hf-donor"
    make_hf(source, modules)
    make_hf(donor, modules, donor=True)
    passed = []
    compute_dtype = "float64" if encoder == "numpy" else "float32"
    encoder_settings = dict(encoder=encoder, device=device, compute_dtype=compute_dtype)
    kwargs = dict(llama_cpp_dir=root, **encoder_settings,
                  scale_steps=3, chunk_groups=16, tile_elements=4096,
                  source_revision="synthetic-base-v1")
    with synthetic_tokenizer(modules):
        # Both sides use the same real pinned converter, including V-head
        # permutations, shared-input fusion, BF16 promotion and norm+1.
        base_float, donor_float = work / "base-f16.gguf", work / "donor-f16.gguf"
        stock_float(source, base_float, modules)
        stock_float(donor, donor_float, modules)
        stock = work / "stock-spin8.gguf"
        stock_quantize(base_float, stock, modules, **encoder_settings)
        direct = work / "direct-spin8.gguf"
        direct_report = hf_stream.convert_hf_to_spin8(source, direct, cache_dir=work / "cache", **kwargs)
        compared = compare(stock, direct, modules)
        passed.append("all_direct_hf_payload_bytes_equal_stock_hf_f16_then_spin8")
        direct_tensors = {tensor.name: tensor for tensor in modules["gguf"].GGUFReader(direct).tensors}
        fused_names = {"blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight"}
        assert all(direct_tensors[name].tensor_type == modules["gguf"].GGMLQuantizationType.SPIN8_G128
                   for name in fused_names)
        passed.append("both_canonical_fused_linear_attention_matrices_are_compressed")
        assert direct_report["memory"]["plain_mmap_tensor_count"] >= 12, direct_report["memory"]
        assert not direct_report["provenance"]["full_float_gguf_staged"]
        assert direct_report["provenance"]["no_mtp"]
        assert direct_report["provenance"]["encoder_cuda_executed"] == (encoder == "torch" and device.startswith("cuda"))
        passed.append("plain_bf16_mmap_slices_and_honest_encoder_device_provenance")
        # Re-encoding all tensors must be avoided when verified cache payloads exist.
        encoder_module = modules["reference_codec"] if encoder == "numpy" else hf_stream.importlib.import_module("reference_torch")
        encoder_name = "encode" if encoder == "numpy" else "encode_torch"
        with patch.object(encoder_module, encoder_name, side_effect=AssertionError("unexpected re-encode")):
            resumed = hf_stream.convert_hf_to_spin8(source, work / "resumed.gguf", cache_dir=work / "cache", **kwargs)
        assert resumed["resume"]["hits"] == direct_report["quantized_tensor_count"]
        compare(direct, work / "resumed.gguf", modules)
        passed.append("verified_resume_reuses_every_compressed_tensor")
        direction_path = work / "direction.npy"
        np.save(direction_path, np.random.default_rng(2121).normal(size=128))
        edited = work / "edited-f16.gguf"
        modules["edit_weights"].edit(argparse.Namespace(
            source=base_float, output=edited, math_base=base_float, math_donor=donor_float, alpha=0.25,
            math_include=[], preserve_missing=[], remove_direction=direction_path,
            outtype="f16", tile_elements=4096, temp_dir=None, force=False))
        stock_edited = work / "stock-edited-spin8.gguf"
        stock_quantize(edited, stock_edited, modules, **encoder_settings)
        direct_edited = work / "direct-edited-spin8.gguf"
        edited_report = hf_stream.convert_hf_to_spin8(
            source, direct_edited, cache_dir=work / "edited-cache", donor_dir=donor, alpha=0.25,
            donor_revision="synthetic-donor-v1", direction_path=direction_path, **kwargs)
        compare(stock_edited, direct_edited, modules)
        assert fused_names <= set(edited_report["provenance"]["edited_tensors"])
        passed.append("task_delta_plus_both_projection_axes_equal_stock_edit_and_spin8_bytes")
        assert not edited_report["math_improvement_measured"]
        assert not edited_report["refusal_behavior_measured"]
        assert edited_report["provenance"]["optimizer_steps"] == 0
        passed.append("no_behavioral_improvement_or_semantic_direction_claims")
        # Ensure plain huge embedding chains never evaluate the whole lazy tensor.
        _, specs = hf_stream._capture(source, work / "capture.gguf", modules, no_mtp=True)
        embedding = specs["token_embd.weight"]
        assert embedding.leaf is not None
        stats = {"max_materialized_canonical_tensor_bytes": 0}
        actual = embedding.read((slice(3, 5), slice(None)), modules, 128, stats)
        reference = next(t for t in modules["gguf"].GGUFReader(base_float).tensors if t.name == embedding.name)
        np.testing.assert_array_equal(actual, reference.data[3:5])
        assert stats["max_materialized_canonical_tensor_bytes"] == 0
        passed.append("embedding_slice_reads_before_cast_even_below_full_tensor_memory_limit")
        transformed = next(value for value in specs.values() if value.leaf is None and np.prod(value.shape) > 128)
        try:
            transformed.read(tuple(slice(None) for _ in transformed.shape), modules, 128, stats)
        except MemoryError:
            passed.append("nonstreamable_large_transform_has_explicit_memory_guard")
        else:
            raise AssertionError("materialization limit was ignored")
        # Same cache cannot be used with a different recipe.
        try:
            hf_stream.convert_hf_to_spin8(source, work / "bad-cache.gguf", cache_dir=work / "cache",
                                         include_embedding=False, **kwargs)
        except ValueError as error:
            assert "mismatch" in str(error)
            passed.append("changed_recipe_rejects_stale_cache_before_publication")
        else:
            raise AssertionError("changed recipe accepted the old cache")
        # A changed shard with unchanged filenames must change the fingerprint.
        before = hf_stream._snapshot_identity(donor, "same-revision")
        with (donor / "model.safetensors").open("r+b") as stream:
            stream.seek(-2, 2)
            old = stream.read(1)
            stream.seek(-2, 2)
            stream.write(bytes([old[0] ^ 1]))
        after = hf_stream._snapshot_identity(donor, "same-revision")
        assert before != after
        with (donor / "model.safetensors").open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(old)
        passed.append("source_identity_hashes_actual_shard_bytes")
        manifest = {"schema": "spin8-hf-source-v2", "status": "COMPLETE", "repo": "fixture/donor",
                    "revision": "a" * 40, "weight_payload_downloaded": True,
                    "files": [{"path": item["file"], "size": item["bytes"], "sha256": item["sha256"],
                               "verified": True} for item in before["files"]]}
        core = {"repo": manifest["repo"], "revision": manifest["revision"],
                "files": [{key: item[key] for key in ("path", "size", "sha256")} for item in manifest["files"]]}
        manifest["content_fingerprint"] = hf_stream.hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        manifest_path = donor / "source.json"
        manifest_path.write_text(json.dumps(manifest))
        identity = hf_stream._snapshot_identity(donor, "a" * 40)
        manifest["completed_utc"] = "changed bookkeeping only"
        manifest_path.write_text(json.dumps(manifest))
        assert identity == hf_stream._snapshot_identity(donor, "a" * 40)
        passed.append("complete_download_manifest_verified_without_timestamp_cache_churn")
        with (donor / "model.safetensors").open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(bytes([old[0] ^ 1]))
        try:
            hf_stream._snapshot_identity(donor, "a" * 40)
        except ValueError as error:
            assert "hash/size" in str(error)
            passed.append("actual_downloaded_weights_must_match_source_manifest_hash")
        else:
            raise AssertionError("altered weights accepted with original manifest")
        with (donor / "model.safetensors").open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(old)
        manifest["status"] = "DOWNLOADING"
        manifest_path.write_text(json.dumps(manifest))
        try:
            hf_stream._snapshot_identity(donor, "a" * 40)
        except ValueError as error:
            assert "COMPLETE" in str(error)
            passed.append("incomplete_source_manifest_rejected_before_weight_reads")
        else:
            raise AssertionError("incomplete source manifest accepted")
        manifest_path.unlink()
    try:
        hf_stream.convert_hf_to_spin8(source, work / "no-donor.gguf", cache_dir=work / "no-donor-cache",
                                     alpha=0.2, **kwargs)
    except ValueError as error:
        assert "donor" in str(error)
        passed.append("nonzero_alpha_requires_explicit_donor")
    else:
        raise AssertionError("nonzero alpha without donor accepted")
    if not modules["torch"].cuda.is_available():
        try:
            hf_stream.convert_hf_to_spin8(source, work / "fake-gpu.gguf", llama_cpp_dir=root,
                                         cache_dir=work / "fake-gpu-cache", device="cuda")
        except RuntimeError as error:
            assert "CUDA" in str(error)
            passed.append("cuda_request_fails_explicitly_on_cpu_runtime")
        else:
            raise AssertionError("CUDA silently fell back to CPU")
    return {"status": "PASS", "checks": passed, "checks_passed": len(passed),
            "tensor_count_compared": compared, "python": hf_stream.platform.python_version(),
            "torch": modules["torch"].__version__, "numpy": np.__version__,
            "device": device, "encoder": encoder, "inference_executed": False, "pretrained_quality_measured": False,
            "synthetic_tokenizer_override": "gpt-2 (shared by both paths; fixture only)",
            "direct_report": direct_report, "edited_report": edited_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Keep the tiny HF/GGUF fixtures and cache for inspection")
    parser.add_argument("--encoder", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if any(args.work_dir.iterdir()):
            raise ValueError("--work-dir must be empty; preserve prior validation outputs separately")
        report = checks(args.llama_cpp_dir, args.work_dir, encoder=args.encoder, device=args.device)
    else:
        with tempfile.TemporaryDirectory(prefix="spin8-hf-integration-") as directory:
            report = checks(args.llama_cpp_dir, Path(directory), encoder=args.encoder, device=args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"direct_report", "edited_report"}}, indent=2))


if __name__ == "__main__":
    main()
