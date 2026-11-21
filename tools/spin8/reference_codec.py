#!/usr/bin/env python3
"""Standalone CPU reference for experimental Spin8-T2 at group size 128.

Each eight-dimensional vector uses one uint16 code. Sixteen codes and one
signed FP16 scale encode 128 weights: 34 bytes, or 2.125 bits/weight. The nine
shared FP64 rotation matrices cost another 4608 bytes per bank. Positive
scales select rotated ternary / identity codes; negative scales select the
uniform four-level fallback. H128 is a normalized Sylvester Hadamard.

This file does not implement a CUDA kernel, native llama.cpp tensor type, or
GGUF writer. Its optional NPZ container is explicitly NOT a GGUF file. No LLM
quality or GPU performance result is implied by its synthetic self-check.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

DIM = 8
GROUP = 128
BLOCKS = GROUP // DIM
NFRAMES = 9
RADIX = 3 ** DIM
POW3 = 3 ** np.arange(DIM, dtype=np.int64)
FALLBACK_START = NFRAMES * RADIX
CODEC = "spin8_g128_v1"


def _positive_int(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def hadamard128(values):
    """Apply normalized H128 along the last axis, using float64 butterflies."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim < 1 or result.shape[-1] != GROUP:
        raise ValueError("Hadamard input must have last dimension 128")
    flat = result.reshape(-1, GROUP)
    width = 1
    while width < GROUP:
        view = flat.reshape(-1, GROUP // (2 * width), 2 * width)
        left = view[..., :width].copy()
        right = view[..., width:].copy()
        view[..., :width] = left + right
        view[..., width:] = left - right
        width *= 2
    result /= math.sqrt(GROUP)
    return result


def _multiplication_tensor():
    table = np.zeros((DIM, DIM, DIM), dtype=np.float64)
    for i in range(DIM):
        table[0, i, i] = table[i, 0, i] = 1.
    for i in range(1, DIM):
        table[i, i, 0] = -1.
    for a, b, c in [(1, 2, 3), (1, 4, 5), (1, 7, 6), (2, 4, 6),
                    (2, 5, 7), (3, 4, 7), (3, 6, 5)]:
        for i, j, k in [(a, b, c), (b, c, a), (c, a, b)]:
            table[i, j, k], table[j, i, k] = 1., -1.
    return table


MULT = _multiplication_tensor()


def _octonion(x, y):
    return np.einsum("...i,...j,ijk->...k", x, y, MULT)


def _triality_triple(rng):
    result = np.repeat(np.eye(DIM)[None], 3, axis=0)
    basis = np.eye(DIM)
    for _ in range(4):
        unit = rng.normal(size=DIM)
        unit /= np.linalg.norm(unit)
        left = np.column_stack([_octonion(unit, e) for e in basis])
        right = np.column_stack([_octonion(e, unit) for e in basis])
        result = np.asarray([m @ r for m, r in zip((left @ right, left, right), result)])
    return result


def make_bank(kind="triality", seed=81237):
    """Return nine float64 SO(8) matrices; initial bank is fixed, not learned.

    Three related triples use the convention A(x*y)=(B*x)*(C*y). Four
    octonion factors sample a subfamily; they do not parameterize all Spin(8).
    Haar rotations are provided only as a non-triality comparison control.
    """
    rng = np.random.default_rng(seed)
    if kind == "triality":
        return np.concatenate([_triality_triple(rng) for _ in range(3)])
    if kind == "haar":
        result = []
        for _ in range(NFRAMES):
            q, r = np.linalg.qr(rng.normal(size=(DIM, DIM)))
            q *= np.where(np.diag(r) >= 0, 1., -1.)[None]
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            result.append(q)
        return np.asarray(result)
    raise ValueError("kind must be 'triality' or 'haar'")


def _validate_bank(bank):
    raw = np.asarray(bank)
    if raw.dtype.kind not in "fiu" or raw.shape != (NFRAMES, DIM, DIM):
        raise ValueError("bank must contain nine real 8x8 matrices")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("bank must be finite")
    orth = np.max(np.abs(result @ result.transpose(0, 2, 1) - np.eye(DIM)))
    if orth > 1.e-10 or np.max(np.abs(np.linalg.det(result) - 1.)) > 1.e-10:
        raise ValueError("bank must contain float64 SO(8) rotations (tolerance 1e-10)")
    return result


def _unrank(rank):
    return ((np.asarray(rank, dtype=np.int64)[..., None] // POW3) % 3 - 1).astype(np.int8)


ALL_TERNARY = _unrank(np.arange(RADIX))
IDENTITY_RANKS = np.flatnonzero(np.count_nonzero(ALL_TERNARY, axis=1) != 2)
IDENTITY_INDEX = np.full(RADIX, -1, dtype=np.int64)
IDENTITY_INDEX[IDENTITY_RANKS] = np.arange(len(IDENTITY_RANKS))
VALID_CODES = FALLBACK_START + len(IDENTITY_RANKS)  # 65498; 38 reserved codes


def _rank(coefficients):
    return np.sum((coefficients.astype(np.int64) + 1) * POW3, axis=-1)


def _identity_nearest(blocks, scale):
    raw = np.clip(np.rint(blocks / scale[:, None, None]), -1, 1).astype(np.int8)
    forbidden = np.count_nonzero(raw, axis=-1) == 2
    order = np.argsort(np.abs(blocks), axis=-1)[..., -3:]
    signs = np.where(np.take_along_axis(blocks, order, axis=-1) >= 0, 1, -1)
    three, one = np.zeros_like(raw), np.zeros_like(raw)
    np.put_along_axis(three, order, signs, axis=-1)
    np.put_along_axis(one, order[..., -1:], signs[..., -1:], axis=-1)
    e_one = np.sum((blocks - scale[:, None, None] * one) ** 2, axis=-1)
    e_three = np.sum((blocks - scale[:, None, None] * three) ** 2, axis=-1)
    replacement = np.where((e_three < e_one)[..., None], three, one)
    return np.where(forbidden[..., None], replacement, raw)


def _scale_grid(transformed, max_level, steps):
    cap = np.linalg.norm(transformed.reshape(-1, BLOCKS, DIM), axis=-1).max(axis=1)
    for factor in np.linspace(.125, 1.05, steps):
        candidate = cap * (factor / max_level)
        if not np.isfinite(candidate).all() or np.any(candidate > np.finfo(np.float16).max):
            raise ValueError("Scale search exceeds the finite FP16 range")
        yield np.maximum(candidate.astype(np.float16).astype(np.float64), 2. ** -24)


def _decode_groups(words, scales, bank):
    """Decode already-validated groups to the canonical float32 output."""
    integer_words = words.astype(np.int64)
    uniform = np.signbit(scales)
    ternary_words = integer_words.copy()
    ternary_words[uniform] = 0
    identity = ternary_words >= FALLBACK_START
    frames = np.where(identity, NFRAMES, ternary_words // RADIX)
    ranks = ternary_words % RADIX
    ranks[identity] = IDENTITY_RANKS[ternary_words[identity] - FALLBACK_START]
    coefficients = _unrank(ranks).astype(np.float64)
    frames[uniform] = NFRAMES
    coefficients[uniform] = ((integer_words[uniform, :, None] >> (2 * np.arange(DIM))) & 3) - 1.5
    extended_bank = np.concatenate([bank, np.eye(DIM)[None]])
    restored = np.einsum("gbo,gboi->gbi", coefficients, extended_bank[frames], optimize=True)
    restored *= np.abs(scales.astype(np.float64))[:, None, None]
    return hadamard128(restored.reshape(-1, GROUP)).astype(np.float32)


def _uniform_chunk(groups, transformed, steps):
    best_error = np.full(len(groups), np.inf)
    best_coeff = np.zeros_like(transformed)
    best_scale = np.ones(len(groups))
    for scale in _scale_grid(transformed, 1.5, steps):
        q = np.clip(np.rint(transformed / scale[:, None] + 1.5), 0, 3) - 1.5
        error = np.sum((transformed - scale[:, None] * q) ** 2, axis=-1)
        improve = error < best_error
        best_error[improve] = error[improve]
        best_coeff[improve], best_scale[improve] = q[improve], scale[improve]
    codes = (best_coeff + 1.5).astype(np.int64).reshape(-1, BLOCKS, DIM)
    words = np.sum(codes << (2 * np.arange(DIM)), axis=-1).astype("<u2")
    return words, -best_scale.astype("<f2")


def _encode_chunk(groups, bank, steps):
    transformed = hadamard128(groups)
    blocks = transformed.reshape(-1, BLOCKS, DIM)
    rotated = np.einsum("gbi,roi->gbro", blocks, bank, optimize=True)
    best_error = np.full(len(groups), np.inf)
    best_scale = np.ones(len(groups))
    best_frame = np.zeros((len(groups), BLOCKS), dtype=np.uint8)
    best_coeff = np.zeros_like(blocks, dtype=np.int8)
    gi, bi = np.arange(len(groups))[:, None], np.arange(BLOCKS)[None]
    for scale in _scale_grid(transformed, 1., steps):
        q = np.clip(np.rint(rotated / scale[:, None, None, None]), -1, 1).astype(np.int8)
        error = np.sum((rotated - scale[:, None, None, None] * q) ** 2, axis=-1)
        frame = error.argmin(axis=-1)
        chosen_error, chosen_q = error[gi, bi, frame], q[gi, bi, frame]
        extra_q = _identity_nearest(blocks, scale)
        extra_error = np.sum((blocks - scale[:, None, None] * extra_q) ** 2, axis=-1)
        prefer = extra_error < chosen_error
        frame = np.where(prefer, NFRAMES, frame)
        chosen_q = np.where(prefer[..., None], extra_q, chosen_q)
        group_error = np.minimum(chosen_error, extra_error).sum(axis=-1)
        improve = group_error < best_error
        best_error[improve], best_scale[improve] = group_error[improve], scale[improve]
        best_frame[improve], best_coeff[improve] = frame[improve], chosen_q[improve]
    ranks = _rank(best_coeff)
    words = best_frame.astype(np.int64) * RADIX + ranks
    special = best_frame == NFRAMES
    words[special] = FALLBACK_START + IDENTITY_INDEX[ranks[special]]
    words, scales = words.astype("<u2"), best_scale.astype("<f2")
    q2_words, q2_scales = _uniform_chunk(groups, transformed, steps)
    # Compare the ACTUAL float32 decoder in original coordinates. This retains
    # the groupwise non-worsening guarantee vs this implemented Q2 encoder.
    ternary_recon = _decode_groups(words, scales, bank).astype(np.float64)
    uniform_recon = _decode_groups(q2_words, q2_scales, bank).astype(np.float64)
    use_uniform = np.sum((groups - uniform_recon) ** 2, axis=-1) < np.sum((groups - ternary_recon) ** 2, axis=-1)
    words[use_uniform], scales[use_uniform] = q2_words[use_uniform], q2_scales[use_uniform]
    return words, scales


def encode(matrix, bank=None, scale_steps=17, chunk_groups=256):
    """Quantize W[out,in]; each row width must be a positive multiple of 128.

    The reference is CPU-only. It processes one chunk at a time and keeps no
    nine-frame expansion of the full matrix. The input matrix itself remains
    caller-owned; a model exporter must stream model tensors separately.
    """
    steps = _positive_int(scale_steps, "scale_steps")
    chunk = _positive_int(chunk_groups, "chunk_groups")
    array = np.asarray(matrix)
    if array.dtype.kind not in "fiu" or array.ndim != 2 or 0 in array.shape or array.shape[1] % GROUP:
        raise ValueError("matrix must be a nonempty real 2D [out,in] array with in divisible by 128")
    frames = _validate_bank(make_bank() if bank is None else bank).copy()
    groups_per_row = array.shape[1] // GROUP
    count = array.shape[0] * groups_per_row
    # Do not silently copy an entire strided tensor just to flatten it.
    grouped = array.reshape(-1, GROUP) if array.flags.c_contiguous else None
    words = np.empty((count, BLOCKS), dtype="<u2")
    scales = np.empty(count, dtype="<f2")
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        if grouped is not None:
            values = np.asarray(grouped[begin:end], dtype=np.float64)
        else:
            index = np.arange(begin, end)
            rows = (index // groups_per_row)[:, None]
            columns = ((index % groups_per_row) * GROUP)[:, None] + np.arange(GROUP)[None]
            values = np.asarray(array[rows, columns], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("matrix must be finite")
        words[begin:end], scales[begin:end] = _encode_chunk(values, frames, steps)
    return {"codec": CODEC, "words": words, "scales": scales, "bank": frames,
            "shape": tuple(int(n) for n in array.shape), "group_size": GROUP,
            "hadamard": True, "identity_fallback": True, "uniform_fallback": True,
            "scale_steps": steps}


def _validate_packed(packed):
    if not isinstance(packed, dict):
        raise ValueError("packed must be a dictionary")
    if packed.get("group_size") != GROUP:
        raise ValueError("Only group size 128 is supported")
    if any(packed.get(k) is not True for k in ("hadamard", "identity_fallback", "uniform_fallback")):
        raise ValueError("This codec requires H128 and both fallback modes")
    if packed.get("codec", CODEC) != CODEC:
        raise ValueError("Unsupported codec version")
    shape = packed.get("shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must be a two-element sequence")
    shape = tuple(_positive_int(x, "shape dimension") for x in shape)
    if shape[1] % GROUP:
        raise ValueError("Each matrix row must have a multiple of 128 columns")
    count = math.prod(shape) // GROUP
    words, scales = np.asarray(packed.get("words")), np.asarray(packed.get("scales"))
    if words.dtype.kind != "u" or words.dtype.itemsize != 2 or words.shape != (count, BLOCKS):
        raise ValueError("words must be uint16 with shape [groups,16]")
    if scales.dtype.kind != "f" or scales.dtype.itemsize != 2 or scales.shape != (count,):
        raise ValueError("scales must be float16 with shape [groups]")
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError("scales must be finite and nonzero; positive and negative zero are invalid")
    ternary = ~np.signbit(scales)
    if np.any(words[ternary] >= VALID_CODES):
        raise ValueError("Reserved ternary codeword (65498 through 65535)")
    bank = _validate_bank(packed.get("bank"))
    return words, scales, bank, shape


def decode(packed):
    """Return the decoded float32 matrix in its original [out,in] shape."""
    words, scales, bank, shape = _validate_packed(packed)
    result = np.empty((len(scales), GROUP), dtype=np.float32)
    for begin in range(0, len(scales), 256):
        end = min(begin + 256, len(scales))
        result[begin:end] = _decode_groups(words[begin:end], scales[begin:end], bank)
    return result.reshape(shape)


def matmul(packed, x):
    """Linear-layer convention: x[...,in] -> result[...,out] = x @ W.T.

    A vector [in] returns W @ x. Only a tile of weight rows is decoded at a
    time. This validates numerical behavior; it is not a fast CUDA GEMM.
    """
    words, scales, bank, shape = _validate_packed(packed)
    raw = np.asarray(x)
    if raw.dtype.kind not in "fiu" or raw.ndim < 1 or raw.shape[-1] != shape[1]:
        raise ValueError("x must be real with last dimension equal to in_features")
    with np.errstate(over="ignore", invalid="ignore"):
        activations = raw.astype(np.float32)
    if not np.isfinite(activations).all():
        raise ValueError("x must be finite and representable in float32")
    flat = activations.reshape(-1, shape[1])
    output = np.empty((len(flat), shape[0]), dtype=np.float32)
    groups_per_row = shape[1] // GROUP
    row_chunk = max(1, 256 // groups_per_row)
    for row in range(0, shape[0], row_chunk):
        row_end = min(row + row_chunk, shape[0])
        begin, end = row * groups_per_row, row_end * groups_per_row
        tile = _decode_groups(words[begin:end], scales[begin:end], bank).reshape(row_end-row, shape[1])
        output[:, row:row_end] = flat @ tile.T
    return output.reshape(activations.shape[:-1] + (shape[0],))


def save(packed, path):
    """Save an NPZ reference container, never a GGUF masquerading by suffix."""
    words, scales, bank, shape = _validate_packed(packed)
    if Path(path).suffix.lower() == ".gguf":
        raise ValueError("save() writes NPZ; use the separate Spin8 GGUF writer for a .gguf file")
    metadata = {k: packed[k] for k in ("group_size", "hadamard", "identity_fallback", "uniform_fallback")}
    metadata.update(codec=CODEC, shape=list(shape), scale_steps=packed.get("scale_steps"))
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, words=words, scales=scales, bank=bank,
                            metadata=np.asarray(json.dumps(metadata, separators=(",", ":"))))


def load(path):
    """Load a reference NPZ without pickle and validate before decoding."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        result = dict(metadata, words=data["words"].copy(), scales=data["scales"].copy(), bank=data["bank"].copy())
    result["shape"] = tuple(result["shape"])
    _validate_packed(result)
    return result


def self_check():
    rng = np.random.default_rng(20261120)
    bank = make_bank()
    weights = rng.normal(size=(7, 384)).astype(np.float32)
    weights[1, 0], weights[2] = 30., 0.
    packed = encode(weights, bank, chunk_groups=2)
    other = encode(weights, bank, chunk_groups=256)
    assert np.array_equal(packed["words"], other["words"])
    assert np.array_equal(packed["scales"], other["scales"])
    strided = encode(np.asfortranarray(weights), bank, chunk_groups=2)
    assert np.array_equal(packed["words"], strided["words"])
    assert np.array_equal(packed["scales"], strided["scales"])
    restored = decode(packed)
    assert restored.dtype == np.float32 and restored.shape == weights.shape
    assert np.array_equal(restored[2], np.zeros(384, dtype=np.float32))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.s8g128.npz"
        save(packed, path)
        loaded = load(path)
        assert np.array_equal(loaded["words"], packed["words"])
        assert np.array_equal(loaded["scales"], packed["scales"])
        assert np.array_equal(decode(loaded), restored)
        file_bytes = path.stat().st_size
    activation = rng.normal(size=(3, 384)).astype(np.float32)
    calculated = matmul(packed, activation)
    expected = activation @ restored.T
    np.testing.assert_allclose(calculated, expected, rtol=2.e-5, atol=2.e-5)
    np.testing.assert_allclose(matmul(packed, activation[0]), restored @ activation[0], rtol=2.e-5, atol=2.e-5)
    groups = weights.astype(np.float64).reshape(-1, GROUP)
    q_words, q_scales = _uniform_chunk(groups, hadamard128(groups), 17)
    baseline = _decode_groups(q_words, q_scales, bank)
    squared_error = np.sum((groups-restored.reshape(-1, GROUP))**2, axis=-1)
    baseline_error = np.sum((groups-baseline)**2, axis=-1)
    assert np.all(squared_error <= baseline_error + 1.e-12)
    x, y = rng.normal(size=(2, 17, 8))
    triple_error = max(float(np.max(np.abs(_octonion(x, y) @ a.T - _octonion(x @ b.T, y @ c.T))))
                       for a, b, c in bank.reshape(3, 3, 8, 8))
    assert triple_error < 1.e-11
    rejected = []
    for label, key, value in [
        ("positive_zero_scale", "scales", np.zeros_like(packed["scales"])),
        ("negative_zero_scale", "scales", np.full_like(packed["scales"], -0.)),
        ("nan_scale", "scales", np.full_like(packed["scales"], np.nan)),
        ("infinite_scale", "scales", np.full_like(packed["scales"], np.inf)),
        ("invalid_shape", "shape", (7, 383)),
        ("nonorthogonal_bank", "bank", bank * 1.01),
    ]:
        broken = dict(packed); broken[key] = value
        try:
            decode(broken)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f"Accepted {label}")
    broken = dict(packed, words=packed["words"].copy(), scales=np.abs(packed["scales"]))
    broken["words"][0, 0] = 65535
    try:
        decode(broken)
    except ValueError:
        rejected.append("reserved_ternary_code")
    else:
        raise AssertionError("Accepted reserved ternary code")
    # The same all-ones word is valid in the uniform mode.
    broken["scales"][0] *= -1
    assert np.isfinite(decode(broken)).all()
    payload = packed["words"].nbytes + packed["scales"].nbytes
    assert payload == len(packed["scales"]) * 34
    return {"codec": CODEC, "scope": "synthetic CPU reference only; no LLM/GPU benchmark",
            "shape": list(weights.shape), "groups": len(packed["scales"]),
            "payload_bytes": payload, "payload_bits_per_weight": payload*8/weights.size,
            "shared_bank_bytes": bank.nbytes, "npz_sample_bytes": file_bytes,
            "nmse": float(squared_error.sum()/np.sum(groups**2)),
            "uniform_baseline_nmse": float(baseline_error.sum()/np.sum(groups**2)),
            "uniform_group_fraction": float(np.mean(np.signbit(packed["scales"]))),
            "matmul_max_abs_difference": float(np.max(np.abs(calculated-expected))),
            "triality_max_abs_error": triple_error,
            "chunking_codes_and_scales_equal": True, "strided_input_codes_and_scales_equal": True,
            "file_roundtrip_equal": True,
            "rejected_invalid_cases": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.self_check:
        parser.error("Use --self-check [--report path.json]")
    report = self_check()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
