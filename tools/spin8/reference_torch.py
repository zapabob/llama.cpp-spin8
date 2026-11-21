"""Experimental PyTorch implementation of the Spin8-T2 g128 reference codec.

This is a tiled encoder and a linear-operator reference, not a llama.cpp CUDA
backend or a full Qwen runtime. CPU tests can validate algebra; A100 memory,
speed, and LLM quality must be measured in Colab.
"""
from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import reference_codec as ref

GROUP = 128
DIM = 8
RADIX = 6561
FRAMES = 9
FALLBACK_START = FRAMES * RADIX


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester transform along the last, power-of-two dimension."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    y = x
    stride = 1
    while stride < n:
        z = y.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        stride *= 2
    return y / math.sqrt(n)


def _tables(device):
    pow3 = 3 ** torch.arange(8, dtype=torch.int64, device=device)
    ranks = torch.arange(RADIX, dtype=torch.int64, device=device)
    ternary = (ranks[:, None] // pow3) % 3 - 1
    dense = ranks[(ternary != 0).sum(-1) != 2]
    inverse = torch.full((RADIX,), -1, dtype=torch.int64, device=device)
    inverse[dense] = torch.arange(len(dense), device=device)
    return pow3, dense, inverse


def _scale(cap, factor, maximum):
    raw = cap * (factor / maximum)
    if not bool(torch.isfinite(raw).all()) or bool((raw > 65504).any()):
        raise ValueError("Scale exceeds finite FP16 range")
    return raw.to(torch.float16).to(cap.dtype).clamp_min(2.0 ** -24)


def _fallback(blocks, scale):
    t = (blocks / scale[:, None, None]).round().clamp(-1, 1)
    forbidden = (t != 0).sum(-1) == 2
    order = blocks.abs().argsort(dim=-1)[..., -3:]
    selected = blocks.gather(-1, order)
    signs = torch.where(selected >= 0, torch.ones_like(selected), -torch.ones_like(selected))
    promoted = torch.zeros_like(t).scatter(-1, order, signs)
    reduced = torch.zeros_like(t).scatter(-1, order[..., -1:], signs[..., -1:])
    se = scale[:, None, None]
    er = ((blocks - se * reduced) ** 2).sum(-1)
    ep = ((blocks - se * promoted) ** 2).sum(-1)
    replacement = torch.where((ep < er)[..., None], promoted, reduced)
    return torch.where(forbidden[..., None], replacement, t)


@torch.no_grad()
def _encode_tile(groups, bank, scale_steps):
    pow3, _, inverse = _tables(groups.device)
    y = fwht(groups)
    blocks = y.reshape(-1, GROUP // DIM, DIM)
    z = torch.einsum("gbi,roi->gbro", blocks, bank)
    cap = torch.linalg.vector_norm(blocks, dim=-1).amax(dim=-1)
    ng = groups.shape[0]
    best_error = torch.full((ng,), torch.inf, dtype=groups.dtype, device=groups.device)
    best_scale = torch.zeros_like(cap)
    best_words = torch.zeros((ng, GROUP // DIM), dtype=torch.int64, device=groups.device)
    gi = torch.arange(ng, device=groups.device)[:, None]
    bi = torch.arange(GROUP // DIM, device=groups.device)[None, :]
    factors = np.linspace(0.125, 1.05, scale_steps)
    for factor in factors:
        scale = _scale(cap, float(factor), 1.0)
        t = (z / scale[:, None, None, None]).round().clamp(-1, 1)
        errors = ((z - scale[:, None, None, None] * t) ** 2).sum(-1)
        frame = errors.argmin(-1)
        selected = t[gi, bi, frame]
        error = errors[gi, bi, frame]
        ident = _fallback(blocks, scale)
        ident_error = ((blocks - scale[:, None, None] * ident) ** 2).sum(-1)
        prefer = ident_error < error
        selected = torch.where(prefer[..., None], ident, selected)
        frame = torch.where(prefer, FRAMES, frame)
        error = torch.minimum(error, ident_error).sum(-1)
        rank = ((selected.to(torch.int64) + 1) * pow3).sum(-1)
        words = torch.where(frame == FRAMES, FALLBACK_START + inverse[rank], frame * RADIX + rank)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_scale = torch.where(improve, scale, best_scale)
        best_words = torch.where(improve[:, None], words, best_words)
    # Search Q2 independently. The caller compares actual float32 decoded
    # candidates in original coordinates, as in the canonical NumPy codec.
    q2_error = torch.full_like(best_error, torch.inf)
    q2_scale = torch.zeros_like(best_scale)
    q2_words = torch.zeros_like(best_words)
    shifts = 2 * torch.arange(DIM, device=groups.device)
    for factor in factors:
        scale = _scale(cap, float(factor), 1.5)
        q = (y / scale[:, None] + 1.5).round().clamp(0, 3)
        error = ((y - scale[:, None] * (q - 1.5)) ** 2).sum(-1)
        words = (q.to(torch.int64).reshape(-1, GROUP // DIM, DIM) << shifts).sum(-1)
        improve = error < q2_error
        q2_error = torch.where(improve, error, q2_error)
        q2_scale = torch.where(improve, -scale, q2_scale)
        q2_words = torch.where(improve[:, None], words, q2_words)
    return best_words, best_scale.to(torch.float16), q2_words, q2_scale.to(torch.float16)


def _select_mode(original, words, scales, q2_words, q2_scales, bank):
    """Bounded CPU verification against the canonical float32 decoder.

    This exact comparison retains a per-group MSE guarantee versus this
    encoder's own Q2 candidate, including when candidates were found in FP32.
    FP32 search need not choose the same Q2 candidate as NumPy FP64 search.
    """
    ternary = ref._decode_groups(words, scales, bank).astype(np.float64)
    uniform = ref._decode_groups(q2_words, q2_scales, bank).astype(np.float64)
    prefer = ((original - uniform) ** 2).sum(-1) < ((original - ternary) ** 2).sum(-1)
    words[prefer], scales[prefer] = q2_words[prefer], q2_scales[prefer]
    return words, scales


@torch.no_grad()
def encode_torch(weights, bank=None, *, scale_steps=17, chunk_groups=256,
                 device=None, compute_dtype=torch.float32, progress: Callable | None = None):
    """Encode one matrix, retaining only one candidate tile on the GPU.

    The returned payload lives on the CPU. FP64 compute can be used for an
    independent CPU-reference comparison. Normal Colab encoding uses FP32;
    differences at rounding boundaries are expected and must be measured.
    Candidate search can run on CUDA; the final hybrid-mode comparison uses
    the canonical NumPy decoder on one CPU tile. This is not a GPU-only or
    production-speed implementation.
    """
    scale_steps = _integer(scale_steps, "scale_steps")
    chunk_groups = _integer(chunk_groups, "chunk_groups")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64")
    bank = ref._validate_bank(ref.make_bank("triality") if bank is None else bank)
    w = weights.detach() if isinstance(weights, torch.Tensor) else None
    array = None if w is not None else np.asarray(weights)
    if w is not None:
        invalid = w.is_complex() or w.dtype == torch.bool or w.ndim != 2 or min(w.shape) < 1 or w.shape[-1] % GROUP
        shape = tuple(w.shape)
    else:
        invalid = array.dtype.kind not in "fiu" or array.ndim != 2 or min(array.shape) < 1 or array.shape[-1] % GROUP
        shape = tuple(array.shape)
    if invalid:
        raise ValueError("weight[out,in] must have nonempty rows of a multiple of 128")
    if device is None:
        device = w.device if w is not None else "cpu"
    device = torch.device(device)
    groups_per_row = shape[1] // GROUP
    group_count = shape[0] * groups_per_row
    groups = w.view(-1, GROUP) if w is not None and w.is_contiguous() else None
    array_groups = array.reshape(-1, GROUP) if array is not None and array.flags.c_contiguous else None
    b = torch.tensor(bank, dtype=compute_dtype, device=device)
    words = np.empty((group_count, GROUP // DIM), dtype=np.uint16)
    scales = np.empty(group_count, dtype=np.float16)
    for start in range(0, group_count, chunk_groups):
        end = min(start + chunk_groups, group_count)
        if array is not None:
            if array_groups is not None:
                values = array_groups[start:end]
            else:
                index = np.arange(start, end)
                values = array[(index // groups_per_row)[:, None],
                               ((index % groups_per_row) * GROUP)[:, None] + np.arange(GROUP)[None]]
            source = torch.from_numpy(np.array(values, copy=True))
        elif groups is None:
            index = torch.arange(start, end, device=w.device)
            rows = (index // groups_per_row)[:, None]
            columns = ((index % groups_per_row) * GROUP)[:, None] + torch.arange(GROUP, device=w.device)[None]
            source = w[rows, columns]
        else:
            source = groups[start:end]
        tile = source.to(device=device, dtype=compute_dtype)
        if not bool(torch.isfinite(tile).all()):
            raise ValueError("Input contains nonfinite values")
        c, s, qc, qs = _encode_tile(tile, b, scale_steps)
        candidate = c.cpu().numpy().astype(np.uint16)
        candidate_scale = s.cpu().numpy()
        q2_candidate = qc.cpu().numpy().astype(np.uint16)
        q2_scale = qs.cpu().numpy()
        original = source.to(device="cpu", dtype=torch.float64).numpy()
        words[start:end], scales[start:end] = _select_mode(
            original, candidate, candidate_scale, q2_candidate, q2_scale, bank)
        if progress:
            progress(end, group_count)
    return {"codec": ref.CODEC, "words": words, "scales": scales, "bank": bank.copy(), "shape": shape,
            "group_size": GROUP, "hadamard": True, "identity_fallback": True,
            "uniform_fallback": True, "scale_steps": scale_steps}


def _coefficients(packed, device, dtype):
    words = torch.as_tensor(np.asarray(packed["words"]).astype(np.int64), device=device)
    scales = torch.as_tensor(np.asarray(packed["scales"]).astype(np.float64), dtype=dtype, device=device)
    uniform = scales < 0
    safe = torch.where(uniform[:, None], 0, words)
    if bool((safe >= 65498).any()):
        raise ValueError("Reserved Spin8 ternary codeword")
    pow3, dense, _ = _tables(device)
    special = safe >= FALLBACK_START
    frame = torch.where(special, FRAMES, safe // RADIX)
    rank = torch.where(special, dense[(safe - FALLBACK_START).clamp(0, len(dense)-1)], safe % RADIX)
    coeff = ((rank[..., None] // pow3) % 3 - 1).to(dtype)
    uq = ((words[..., None] >> (2 * torch.arange(DIM, device=device))) & 3).to(dtype) - 1.5
    coeff = torch.where(uniform[:, None, None], uq, coeff)
    frame = torch.where(uniform[:, None], FRAMES, frame)
    return frame, coeff, scales.abs()


def decode_torch(packed, *, device="cpu", dtype=torch.float32, bank_tensor=None):
    """Differentiable decode in the bank, with fixed discrete codes/scales.

    The packed payload is validated by the independent CPU codec. A supplied
    bank tensor must contain SO(8) rotations, checked at its storage precision;
    the numerical operations retain its gradient graph.
    """
    ref._validate_packed(packed)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Decoded dtype must be floating point")
    if bank_tensor is not None:
        if not isinstance(bank_tensor, torch.Tensor) or not bank_tensor.is_floating_point() or bank_tensor.shape != (FRAMES, DIM, DIM):
            raise ValueError("bank_tensor must contain nine floating-point 8x8 matrices")
        checked = bank_tensor.detach().to(dtype=torch.float64)
        tolerance = max(1e-10, 32 * torch.finfo(bank_tensor.dtype).eps)
        orth_error = (checked @ checked.transpose(-1, -2) - torch.eye(DIM, dtype=checked.dtype, device=checked.device)).abs().max()
        det_error = (torch.linalg.det(checked) - 1).abs().max()
        if not bool(torch.isfinite(checked).all()) or float(orth_error) > tolerance or float(det_error) > tolerance:
            raise ValueError("bank_tensor must contain SO(8) rotations")
    device = torch.device(device)
    frame, coeff, scale = _coefficients(packed, device, dtype)
    bank = (torch.as_tensor(packed["bank"], dtype=dtype, device=device) if bank_tensor is None
            else bank_tensor.to(device=device, dtype=dtype))
    all_banks = torch.cat((bank, torch.eye(DIM, dtype=dtype, device=device)[None]), dim=0)
    blocks = torch.einsum("gbi,gbij->gbj", coeff, all_banks[frame])
    y = (blocks * scale[:, None, None]).reshape(-1, GROUP)
    return fwht(y).reshape(packed["shape"])


@torch.no_grad()
def linear_torch(packed, x, *, row_tile=64):
    """Tiled dequantize + matmul reference. Not a fused low-bit CUDA GEMM."""
    _, _, _, (rows, width) = ref._validate_packed(packed)
    row_tile = _integer(row_tile, "row_tile")
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 1 or x.shape[-1] != width or not bool(torch.isfinite(x).all()):
        raise ValueError("x must be a finite floating tensor with last dimension in_features")
    groups_per_row = width // GROUP
    result = torch.empty((*x.shape[:-1], rows), dtype=x.dtype, device=x.device)
    for start in range(0, rows, row_tile):
        end = min(rows, start + row_tile)
        lo, hi = start * groups_per_row, end * groups_per_row
        part = dict(packed, words=packed["words"][lo:hi], scales=packed["scales"][lo:hi],
                    shape=(end-start, width))
        decoded = decode_torch(part, device=x.device, dtype=x.dtype)
        result[..., start:end] = x @ decoded.T
    return result


def fit_triality_bank(weights, *, initial_bank=None, steps=20, lr=0.03, scale_steps=9,
                      chunk_groups=256, activations=None, checkpoint: Callable | None = None):
    """Alternating fixed-code bank optimization, selecting by measured loss.

    Uses the 84-parameter full Lie-algebra family (three related triples).
    No STE is used for the codes. Their Euclidean selection is a heuristic
    for activation loss, so the returned bank is the best measured reencoded
    candidate. The optimizer may explore worse candidates; the original bank
    is always retained as a candidate for the returned result.
    This fits a small matrix sample, not all 27B weights at once.
    """
    from triality_lie import build_triality_lie_basis
    steps = _integer(steps, "steps", minimum=0)
    scale_steps = _integer(scale_steps, "scale_steps")
    chunk_groups = _integer(chunk_groups, "chunk_groups")
    if isinstance(lr, (bool, np.bool_)) or not isinstance(lr, numbers.Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid learning configuration")
    raw = weights.detach() if isinstance(weights, torch.Tensor) else torch.as_tensor(weights)
    if raw.is_complex() or raw.dtype == torch.bool or raw.ndim != 2 or min(raw.shape) < 1 or raw.shape[-1] % GROUP:
        raise ValueError("weights must be nonempty real [out,in] with in_features divisible by 128")
    w = raw.to(dtype=torch.float32)
    if not bool(torch.isfinite(w).all()):
        raise ValueError("weights must be finite and representable in float32")
    device = w.device
    initial = ref._validate_bank(ref.make_bank("triality") if initial_bank is None else initial_bank)
    a, b, c = np.moveaxis(initial.reshape(3, 3, DIM, DIM), 1, 0)
    related_error = np.max(np.abs(np.einsum("nkl,ijl->nijk", a, ref.MULT)
                                 - np.einsum("npi,nqj,pqk->nijk", b, c, ref.MULT)))
    if related_error > 1e-9:
        raise ValueError("initial_bank must be three related triality triples")
    basis = torch.as_tensor(build_triality_lie_basis(ref.MULT), dtype=torch.float64, device=device)
    base = torch.as_tensor(initial, dtype=torch.float64, device=device).reshape(3, 3, 8, 8)
    theta = torch.zeros((3, 28), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    x = None
    if activations is not None:
        raw_x = activations.detach() if isinstance(activations, torch.Tensor) else torch.as_tensor(activations)
        if raw_x.is_complex() or raw_x.dtype == torch.bool or raw_x.ndim != 2 or min(raw_x.shape) < 1 or raw_x.shape[-1] != w.shape[-1]:
            raise ValueError("activations must be nonempty real [tokens,in_features]")
        x = raw_x.to(device=device, dtype=w.dtype)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("activations must be finite and representable in float32")
    target = w if x is None else x @ w.T
    denominator = target.square().sum().clamp_min(torch.finfo(w.dtype).tiny)

    def bank_now():
        generators = torch.einsum("na,raij->nrij", theta, basis) / math.sqrt(3.0)
        return (torch.matrix_exp(generators) @ base).reshape(9, 8, 8)

    def objective(decoded):
        predicted = decoded if x is None else x @ decoded.T
        return (predicted - target).square().sum() / denominator

    best_loss = float("inf")
    best_bank = initial.copy()
    history = []
    for step in range(steps + 1):
        bank = bank_now()
        current = bank.detach().cpu().numpy()
        packed = encode_torch(w, current, scale_steps=scale_steps, chunk_groups=chunk_groups,
                              device=device)
        reconstructed = decode_torch(packed, device=device, bank_tensor=bank)
        loss = objective(reconstructed)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite training loss")
        if value < best_loss:
            best_loss, best_bank = value, current.copy()
        history.append({"step": step, "loss": value, "best_loss": best_loss})
        if checkpoint:
            checkpoint(step, {"theta": theta.detach().cpu().numpy(), "best_bank": best_bank,
                               "history": history, "optimizer": optimizer.state_dict()})
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if theta.grad is None or not bool(torch.isfinite(theta.grad).all()):
                raise RuntimeError("Triality parameters have nonfinite or missing gradients")
            history[-1]["gradient_norm"] = float(theta.grad.norm().detach())
            history[-1]["finite_gradients"] = True
            optimizer.step()
    return best_bank, {"objective": "weight_nmse" if x is None else "activation_output_nmse",
                       "parameters": 84, "steps": steps, "history": history,
                       "best_training_loss": best_loss, "full_model_evaluation": False}


def self_test():
    """Small deterministic CPU checks; no measured GPU or language-model claims."""
    rng = np.random.default_rng(20261120)
    weight = rng.normal(scale=0.08, size=(5, 256))
    bank = ref.make_bank("triality")
    cpu = ref.encode(weight, bank, scale_steps=9)
    pt = encode_torch(torch.tensor(weight, dtype=torch.float64), bank, scale_steps=9,
                      compute_dtype=torch.float64, chunk_groups=3)
    restored = ref.decode(pt)
    decoded = decode_torch(pt, dtype=torch.float64).numpy()
    error = float(np.max(np.abs(restored - decoded)))
    if error > 2e-7:
        raise AssertionError(f"Torch decode mismatch: {error}")
    if not np.array_equal(cpu["words"], pt["words"]) or not np.array_equal(cpu["scales"], pt["scales"]):
        raise AssertionError("FP64 encoding differs from the independent NumPy reference")
    x = torch.tensor(rng.normal(size=(3, 256)), dtype=torch.float64)
    result = linear_torch(pt, x, row_tile=2)
    linear_error = float((result - x @ decode_torch(pt, dtype=torch.float64).T).abs().max())
    if linear_error > 1e-10:
        raise AssertionError("Tiled linear operator differs from dense reconstruction")
    zero = encode_torch(torch.zeros((1, 128), dtype=torch.float64), compute_dtype=torch.float64)
    if not np.array_equal(ref.decode(zero), np.zeros((1, 128))):
        raise AssertionError("Zero weights did not survive encoding")

    # Strided inputs must preserve the result without forcing a whole-matrix
    # contiguous copy in the encoder.
    strided = torch.tensor(weight.T.copy(), dtype=torch.float64).T
    assert not strided.is_contiguous()
    sp = encode_torch(strided, bank, scale_steps=9, compute_dtype=torch.float64, chunk_groups=2)
    assert np.array_equal(sp["words"], pt["words"])
    assert np.array_equal(sp["scales"], pt["scales"])

    w32 = torch.tensor(weight, dtype=torch.float32)
    p32 = encode_torch(w32, bank, scale_steps=9, chunk_groups=3)
    canonical32 = ref.decode(p32)
    decoded32 = decode_torch(p32).numpy()
    decode32_error = float(np.max(np.abs(canonical32 - decoded32)))
    np.testing.assert_allclose(decoded32, canonical32, rtol=3e-6, atol=2e-7)
    x32 = x.to(dtype=torch.float32)
    linear32 = linear_torch(p32, x32, row_tile=2).numpy()
    expected32 = x32.numpy() @ canonical32.T
    linear32_error = float(np.max(np.abs(linear32 - expected32)))
    np.testing.assert_allclose(linear32, expected32, rtol=3e-5, atol=3e-6)
    # Check the guarantee against the FP32 encoder's own Q2 candidate, not
    # against a differently rounded or exhaustive four-level search.
    _, _, qc, qs = _encode_tile(w32.reshape(-1, GROUP), torch.tensor(bank, dtype=torch.float32), 9)
    q2 = ref._decode_groups(qc.numpy().astype(np.uint16), qs.numpy(), bank)
    original = w32.numpy().astype(np.float64).reshape(-1, GROUP)
    hybrid_error = ((original - canonical32.reshape(-1, GROUP)) ** 2).sum(-1)
    q2_error = ((original - q2) ** 2).sum(-1)
    assert np.all(hybrid_error <= q2_error + 1e-12)

    from triality_lie import default_multiplication_tensor
    assert np.array_equal(default_multiplication_tensor(), ref.MULT)
    calibration = torch.tensor(rng.normal(size=(12, 256)), dtype=torch.float32)
    heldout = torch.tensor(rng.normal(size=(13, 256)), dtype=torch.float32)
    fitted, training = fit_triality_bank(w32[:2], initial_bank=bank, steps=3,
                                         scale_steps=9, chunk_groups=2,
                                         activations=calibration)
    assert training["parameters"] == 84
    gradient_rows = training["history"][:-1]
    assert len(gradient_rows) == 3 and all(row["finite_gradients"] for row in gradient_rows)
    assert any(row["gradient_norm"] > 0 for row in gradient_rows)
    assert training["best_training_loss"] <= training["history"][0]["loss"]
    initial_packed = encode_torch(w32[:2], bank, scale_steps=9, chunk_groups=2)
    fitted_packed = encode_torch(w32[:2], fitted, scale_steps=9, chunk_groups=2)
    heldout_target = heldout @ w32[:2].T
    heldout_norm = heldout_target.square().sum()
    heldout_initial = float(((linear_torch(initial_packed, heldout) - heldout_target).square().sum() / heldout_norm))
    heldout_fitted = float(((linear_torch(fitted_packed, heldout) - heldout_target).square().sum() / heldout_norm))
    assert math.isfinite(heldout_initial) and math.isfinite(heldout_fitted)
    a, b, c = np.moveaxis(fitted.reshape(3, 3, DIM, DIM), 1, 0)
    triality_error = float(np.max(np.abs(np.einsum("nkl,ijl->nijk", a, ref.MULT)
                                        - np.einsum("npi,nqj,pqk->nijk", b, c, ref.MULT))))
    assert triality_error < 1e-9

    rejected = []
    reflection = bank.copy()
    reflection[0, 0] *= -1
    cases = [
        ("reflection_bank", lambda: encode_torch(w32, reflection)),
        ("bool_scale_steps", lambda: encode_torch(w32, scale_steps=True)),
        ("bool_chunk_groups", lambda: encode_torch(w32, chunk_groups=True)),
        ("bool_row_tile", lambda: linear_torch(p32, x32, row_tile=True)),
        ("bool_training_steps", lambda: fit_triality_bank(w32[:2], steps=True)),
        ("bool_learning_rate", lambda: fit_triality_bank(w32[:2], steps=0, lr=True)),
        ("complex_weights", lambda: encode_torch(w32.to(dtype=torch.complex64))),
        ("positive_zero_scale", lambda: decode_torch(dict(p32, scales=np.zeros_like(p32["scales"])))),
        ("negative_zero_scale", lambda: decode_torch(dict(p32, scales=np.full_like(p32["scales"], -0.)))),
    ]
    for label, operation in cases:
        try:
            operation()
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f"Accepted {label}")
    return {"scope": "synthetic CPU reference checks only; no LLM or GPU benchmark",
            "torch_version": torch.__version__, "device": "cpu", "seed": 20261120,
            "group_size": GROUP, "shape": list(weight.shape),
            "fp64_codewords_match_numpy": True, "fp64_scales_match_numpy": True,
            "decode_fp64_vs_canonical_fp32_max_abs": error,
            "linear_fp64_tiled_vs_dense_max_abs": linear_error,
            "strided_codes_and_scales_equal": True,
            "decode_fp32_vs_canonical_fp32_max_abs": decode32_error,
            "linear_fp32_vs_canonical_max_abs": linear32_error,
            "fp32_hybrid_not_worse_than_own_q2_per_group": True,
            "hybrid_minus_q2_group_mse_max": float(np.max((hybrid_error - q2_error) / GROUP)),
            "training_shape": [2, 256], "calibration_tokens": 12, "heldout_tokens": 13,
            "training": training, "fitted_triality_max_abs": triality_error,
            "heldout_initial_output_nmse": heldout_initial,
            "heldout_fitted_output_nmse": heldout_fitted,
            "heldout_is_synthetic_only": True,
            "rejected_invalid_cases": rejected,
            "cuda_tested": False, "a100_tested": False, "full_model_evaluation": False}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run the default CPU self-test")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(self_test(), indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print(report, end="")
