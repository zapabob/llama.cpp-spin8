#!/usr/bin/env python3
"""Create a deterministic four-layer Qwen35 graph fixture with synthetic weights."""
from __future__ import annotations

from atomic_output import publish_file

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf


def fixture_weights(seed: int = 20261119, *, mtp: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden, intermediate, vocab = 128, 256, 256
    head_dim, head_count, kv_count = 64, 2, 1
    state_dim, key_heads, value_heads, conv_kernel = 32, 2, 4, 4
    key_dim, value_dim = state_dim * key_heads, state_dim * value_heads
    tensors: dict[str, np.ndarray] = {}

    def random(name: str, shape: tuple[int, ...], scale: float = 0.04) -> None:
        tensors[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(name: str, width: int) -> None:
        tensors[name] = np.ones(width, dtype=np.float32)

    random("token_embd.weight", (vocab, hidden), 0.1)
    norm("output_norm.weight", hidden)
    random("output.weight", (vocab, hidden), 0.06)
    for layer in range(5 if mtp else 4):
        prefix = f"blk.{layer}."
        norm(prefix + "attn_norm.weight", hidden)
        norm(prefix + "post_attention_norm.weight", hidden)
        random(prefix + "ffn_gate.weight", (intermediate, hidden))
        random(prefix + "ffn_up.weight", (intermediate, hidden))
        random(prefix + "ffn_down.weight", (hidden, intermediate))
        if layer >= 3:
            random(prefix + "attn_q.weight", (2 * head_dim * head_count, hidden))
            random(prefix + "attn_k.weight", (head_dim * kv_count, hidden))
            random(prefix + "attn_v.weight", (head_dim * kv_count, hidden))
            random(prefix + "attn_output.weight", (hidden, head_dim * head_count))
            norm(prefix + "attn_q_norm.weight", head_dim)
            norm(prefix + "attn_k_norm.weight", head_dim)
        else:
            random(prefix + "attn_qkv.weight", (2 * key_dim + value_dim, hidden))
            random(prefix + "attn_gate.weight", (value_dim, hidden))
            random(prefix + "ssm_beta.weight", (value_heads, hidden))
            random(prefix + "ssm_alpha.weight", (value_heads, hidden))
            random(prefix + "ssm_out.weight", (hidden, value_dim))
            random(prefix + "ssm_conv1d.weight", (2 * key_dim + value_dim, conv_kernel), 0.01)
            tensors[prefix + "ssm_conv1d.weight"][:, -1] += 1.0
            tensors[prefix + "ssm_dt.bias"] = np.full(value_heads, -2.0, dtype=np.float32)
            tensors[prefix + "ssm_a"] = -np.linspace(0.5, 1.0, value_heads, dtype=np.float32)
            norm(prefix + "ssm_norm.weight", state_dim)
        if layer == 4:
            random(prefix + "nextn.eh_proj.weight", (hidden, 2 * hidden))
            norm(prefix + "nextn.enorm.weight", hidden)
            norm(prefix + "nextn.hnorm.weight", hidden)
            random(prefix + "nextn.embed_tokens.weight", (vocab, hidden), 0.1)
            random(prefix + "nextn.shared_head_head.weight", (vocab, hidden), 0.06)
            norm(prefix + "nextn.shared_head_norm.weight", hidden)
    return tensors


def fixture_metadata(writer: gguf.GGUFWriter, tokenizer: str = "none", *, mtp: bool = False) -> None:
    writer.add_name("Synthetic Qwen35 Spin8 runtime fixture")
    writer.add_context_length(128)
    writer.add_embedding_length(128)
    writer.add_block_count(5 if mtp else 4)
    if mtp:
        writer.add_nextn_predict_layers(1)
    writer.add_feed_forward_length(256)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_key_length(64)
    writer.add_value_length(64)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(64)
    writer.add_rope_dimension_sections([8, 8, 16, 0])
    writer.add_rope_freq_base(10000.0)
    writer.add_full_attention_interval(4)
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_inner_size(128)
    writer.add_ssm_state_size(32)
    writer.add_ssm_time_step_rank(4)
    writer.add_ssm_group_count(2)
    writer.add_vocab_size(256)
    if tokenizer == "gpt2-byte":
        from gguf.vocab import bytes_to_unicode
        mapping = bytes_to_unicode()
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        writer.add_token_list([mapping[value] for value in range(256)])
        writer.add_token_types([int(gguf.TokenType.CONTROL)] + [int(gguf.TokenType.NORMAL)] * 255)
        writer.add_key_value("tokenizer.ggml.merges", [], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
        writer.add_bos_token_id(0)
        writer.add_eos_token_id(0)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
    elif tokenizer == "none":
        writer.add_tokenizer_model("none")
    else:
        raise ValueError("Unsupported fixture tokenizer")
    writer.add_string("spin8.fixture.kind", "synthetic_graph_parity")
    writer.add_string("spin8.fixture.tokenizer", "gpt2_byte_with_byte0_reserved_as_bos_eos" if tokenizer == "gpt2-byte" else "dummy_token_ids_only")
    writer.add_bool("spin8.fixture.pretrained", False)


def write_fixture(path: Path, tensors: dict[str, np.ndarray], *, overwrite: bool = False, tokenizer: str = "none", mtp: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    writer = None
    try:
        writer = gguf.GGUFWriter(temporary, "qwen35")
        fixture_metadata(writer, tokenizer, mtp=mtp)
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        for name, tensor in tensors.items():
            writer.add_tensor(name, np.ascontiguousarray(tensor, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        publish_file(temporary, path, overwrite=overwrite)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tokenizer", choices=["none", "gpt2-byte"], default="none")
    parser.add_argument("--mtp", action="store_true", help="Append one NextN layer for default-skip regression tests")
    args = parser.parse_args()
    tensors = fixture_weights(args.seed, mtp=args.mtp)
    output = args.output_dir / "tiny-qwen35-f32.gguf"
    write_fixture(output, tensors, overwrite=args.force, tokenizer=args.tokenizer, mtp=args.mtp)
    report = {
        "artifact": output.name, "architecture": "qwen35", "seed": args.seed,
        "layers": ["gated_delta_net", "gated_delta_net", "gated_delta_net", "full_attention"],
        "nextn_predict_layers": int(args.mtp),
        "pretrained": False, "tokenizer": args.tokenizer, "language_quality_meaningful": False,
        "vocab_size": 256, "prompt_tokens": [1, 4, 7, 9, 13],
        "next_tokens": [17, 21, 25],
        "tensor_count": len(tensors), "parameter_count": sum(t.size for t in tensors.values()),
        "file_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "weights_sha256": {name: hashlib.sha256(tensor.astype("<f4").tobytes()).hexdigest() for name, tensor in tensors.items()},
    }
    (args.output_dir / "fixture.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "weights_sha256"}, indent=2))


if __name__ == "__main__":
    main()
