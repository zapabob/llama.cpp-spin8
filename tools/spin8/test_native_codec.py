"""Independent NumPy checks for native compressed CPU operators."""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_codec as ref


def payload(packed):
    data = np.empty((len(packed["scales"]), 34), dtype=np.uint8)
    data[:, :32] = packed["words"].astype("<u2").view(np.uint8).reshape(-1, 32)
    data[:, 32:] = packed["scales"].astype("<f2").view(np.uint8).reshape(-1, 2)
    return data


def precise_decode(packed):
    words = packed["words"].astype(np.int64)
    uniform = np.signbit(packed["scales"])
    safe = np.where(uniform[:, None], 0, words)
    identity = safe >= 59049
    frames = np.where(identity, 9, safe // 6561)
    ranks = safe % 6561
    ranks[identity] = ref.IDENTITY_RANKS[safe[identity] - 59049]
    coefficients = ref.ALL_TERNARY[ranks].astype(np.float64)
    frames[uniform] = 9
    coefficients[uniform] = ((words[uniform, :, None] >> (2 * np.arange(8))) & 3) - 1.5
    bank = np.concatenate((packed["bank"], np.eye(8)[None]))
    values = np.einsum("gbo,gboi->gbi", coefficients, bank[frames])
    values *= np.abs(packed["scales"].astype(np.float64))[:, None, None]
    h = np.ones((1, 1))
    for _ in range(7):
        h = np.block([[h, h], [h, -h]])
    return (values.reshape(-1, 128) @ (h / np.sqrt(128))).reshape(packed["shape"])


def configure(lib):
    p = C.c_void_p
    lib.spin8_g128_validate_f64.argtypes = [p, C.c_size_t, p, C.c_size_t, C.c_int64, C.c_int64]
    lib.spin8_g128_validate_f64.restype = C.c_int
    lib.spin8_g128_dot_f64.argtypes = [p, p, p, C.c_int64]
    lib.spin8_g128_dot_f64.restype = C.c_double
    lib.spin8_g128_decode_row_f32.argtypes = [p, p, p, C.c_int64]
    lib.spin8_g128_decode_row_f32.restype = None
    lib.spin8_g128_matmul_cpu_f64.argtypes = [p, C.c_size_t, p, p, C.c_size_t, p, C.c_size_t, *([C.c_int64] * 5)]
    lib.spin8_g128_matmul_cpu_f64.restype = C.c_int
    lib.spin8_g128_identity_rank.argtypes = [C.c_int]
    lib.spin8_g128_identity_rank.restype = C.c_int
    lib.spin8_g128_scale_value.argtypes = [C.c_uint16, C.POINTER(C.c_double)]
    lib.spin8_g128_scale_value.restype = C.c_int


def validate(lib, raw, bank, rows, width, size=None):
    return lib.spin8_g128_validate_f64(raw.ctypes.data, raw.nbytes if size is None else size,
                                      bank.ctypes.data, bank.size, rows, width)


def decoded_rows(lib, packed, raw):
    rows, width = packed["shape"]
    out = np.empty((rows, width), dtype=np.float32)
    for row in range(rows):
        lib.spin8_g128_decode_row_f32(raw.ctypes.data + row * (width // 128 * 34),
                                     packed["bank"].ctypes.data, out[row].ctypes.data, width)
    return out


def run(lib):
    configure(lib)
    rng = np.random.default_rng(20261120)
    bank = ref.make_bank()
    ranks = np.asarray([lib.spin8_g128_identity_rank(i) for i in range(6449)])
    np.testing.assert_array_equal(ranks, ref.IDENTITY_RANKS)
    assert lib.spin8_g128_identity_rank(-1) == -1
    assert lib.spin8_g128_identity_rank(6449) == -1
    half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    magnitude = C.c_double()
    valid_scales = 0
    for bits in range(65536):
        status = lib.spin8_g128_scale_value(bits, C.byref(magnitude))
        if np.isfinite(half[bits]) and half[bits] != 0:
            assert status == 0 and magnitude.value == abs(half[bits]), bits
            valid_scales += 1
        else:
            assert status == 3, bits
    codes = {}
    for mode, count, sign in (("ternary", 65498, 1), ("uniform", 65536, -1)):
        groups = (count + 15) // 16
        words = np.zeros(groups * 16, dtype=np.uint16)
        words[:count] = np.arange(count, dtype=np.uint16)
        packed = {"words": words.reshape(groups, 16), "scales": np.full(groups, sign, dtype=np.float16),
                  "bank": bank, "shape": (groups, 128), "group_size": 128,
                  "hadamard": True, "identity_fallback": True, "uniform_fallback": True}
        raw = payload(packed)
        assert validate(lib, raw, bank, groups, 128) == 0
        native = decoded_rows(lib, packed, raw)
        canonical = ref.decode(packed)
        np.testing.assert_allclose(native, canonical, rtol=2e-6, atol=2e-6)
        codes[mode] = {"codewords_checked": count, "decode_max_abs": float(np.max(np.abs(native - canonical)))}
    weights = rng.normal(scale=0.1, size=(7, 384))
    weights[1] = 0
    weights[2, 7] = 3
    packed = ref.encode(weights, bank)
    raw = payload(packed)
    precise = precise_decode(packed)
    native = decoded_rows(lib, packed, raw)
    np.testing.assert_allclose(native, ref.decode(packed), rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(native[1], np.zeros(384, dtype=np.float32))
    x = rng.normal(size=(4, 391))
    output = np.full((4, 10), 123456.0)
    status = lib.spin8_g128_matmul_cpu_f64(raw.ctypes.data, raw.nbytes, bank.ctypes.data,
                                          x.ctypes.data, x.size, output.ctypes.data, output.size,
                                          7, 384, 4, 391, 10)
    assert status == 0
    expected = x[:, :384] @ precise.T
    np.testing.assert_allclose(output[:, :7], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(output[:, 7:], np.full((4, 3), 123456.0))
    row_bytes = 384 // 128 * 34
    x32 = x[:, :384].copy().astype(np.float32)
    row_result = np.asarray([[lib.spin8_g128_dot_f64(raw.ctypes.data + row * row_bytes,
                                                   bank.ctypes.data, token.ctypes.data, 384)
                              for row in range(7)] for token in x32])
    expected_row = x32.astype(np.float64) @ precise.T
    np.testing.assert_allclose(row_result, expected_row, rtol=2e-12, atol=2e-12)
    canonical_linear = x32 @ ref.decode(packed).T
    native_linear = row_result.astype(np.float32)
    np.testing.assert_allclose(native_linear, canonical_linear, rtol=2e-5, atol=2e-6)
    rejected = []
    for label, bits in (("positive_zero_scale", 0), ("negative_zero_scale", 0x8000),
                        ("infinite_scale", 0x7c00), ("nan_scale", 0x7e00)):
        bad = raw.copy()
        bad[0, 32:] = np.asarray([bits], dtype="<u2").view(np.uint8)
        assert validate(lib, bad, bank, 7, 384) == 3
        rejected.append(label)
    for word in range(65498, 65536):
        bad = raw.copy()
        bad[0, :2] = np.asarray([word], dtype="<u2").view(np.uint8)
        bad[0, 33] &= 0x7f
        assert validate(lib, bad, bank, 7, 384) == 4
        bad[0, 33] |= 0x80
        assert validate(lib, bad, bank, 7, 384) == 0
    rejected.append("all_38_reserved_positive_scale_codewords")
    for label, bad_bank in (("nonorthogonal_bank", bank * 1.01),
                            ("nan_bank", np.full_like(bank, np.nan)),
                            ("infinite_bank", np.full_like(bank, np.inf))):
        assert validate(lib, raw, bad_bank, 7, 384) == 5
        rejected.append(label)
    reflection = bank.copy()
    reflection[0, 0] *= -1
    assert validate(lib, raw, reflection, 7, 384) == 5
    rejected.append("reflection_bank")
    assert validate(lib, raw, bank, 7, 383) == 2
    assert validate(lib, raw, bank, 0, 384) == 2
    assert validate(lib, raw, bank, 2**63 - 1, 2**62) == 2
    assert validate(lib, raw, bank, 7, 384, raw.nbytes - 1) == 1
    rejected.extend(("invalid_shape", "zero_rows", "dimension_overflow", "truncated_payload"))
    bad_x = x.copy()
    bad_x[1, 4] = np.nan
    assert lib.spin8_g128_matmul_cpu_f64(raw.ctypes.data, raw.nbytes, bank.ctypes.data,
                                        bad_x.ctypes.data, bad_x.size, output.ctypes.data, output.size,
                                        7, 384, 4, 391, 10) == 6
    rejected.append("nonfinite_activation")
    assert lib.spin8_g128_matmul_cpu_f64(raw.ctypes.data, raw.nbytes, bank.ctypes.data,
                                        x.ctypes.data, 12, output.ctypes.data, output.size,
                                        7, 384, 4, 391, 10) == 1
    rejected.append("short_activation_buffer")
    return {"scope": "native CPU compressed operators; no GPU execution or LLM quality measurement",
            "all_identity_ranks_checked": len(ranks), "all_fp16_patterns_checked": 65536,
            "valid_finite_nonzero_fp16_scales": valid_scales, "codeword_decoding": codes,
            "matrix_shape": [7, 384], "tokens": 4,
            "f64_matmul_max_abs": float(np.max(np.abs(output[:, :7] - expected))),
            "f64_row_dot_f32_activation_max_abs": float(np.max(np.abs(row_result - expected_row))),
            "f32_output_vs_canonical_matmul_max_abs": float(np.max(np.abs(native_linear - canonical_linear))),
            "strided_activation_and_output_checked": True,
            "rejected_invalid_cases": rejected, "cuda_executed": False, "full_model_evaluation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("native-codec-selfcheck.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="spin8-native-check-") as directory:
        library = args.library
        if library is None:
            library = Path(directory) / "libspin8-reference.so"
            subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC",
                            str(ROOT / "ggml/src/ggml-spin8.cpp"),
                            "-I" + str(ROOT / "ggml/include"), "-o", str(library)], check=True)
        report = run(C.CDLL(str(library.resolve())))
    text = json.dumps(report, indent=2) + "\n"
    args.report.write_text(text)
    print(text, end="")
