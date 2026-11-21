"""Run the fused CUDA kernels against the NumPy reference on a CUDA device."""
from __future__ import annotations
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import reference_codec as ref
from test_native_codec import payload

ROOT = Path(__file__).resolve().parents[2]


def configure_runtime(runtime):
    runtime.cudaGetErrorString.argtypes = [C.c_int]
    runtime.cudaGetErrorString.restype = C.c_char_p
    runtime.cudaGetDeviceCount.argtypes = [C.POINTER(C.c_int)]
    runtime.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
    runtime.cudaFree.argtypes = [C.c_void_p]
    runtime.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
    for name in ("cudaGetDeviceCount", "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaDeviceSynchronize"):
        getattr(runtime, name).restype = C.c_int


def check(runtime, code):
    if code:
        raise RuntimeError(runtime.cudaGetErrorString(code).decode())


def run_case(runtime, kernels, packed, seed):
    rng = np.random.default_rng(seed)
    raw = payload(packed)
    canonical = ref.decode(packed)
    rows, width = packed["shape"]
    x = rng.normal(size=(3, width + 5)).astype(np.float32)
    output = np.full((3, rows + 3), 123456.0, dtype=np.float32)
    ids = np.asarray([rows - 1, 0, rows // 2, rows - 1, -1, rows], dtype=np.int32)
    gathered = np.zeros((len(ids), width), dtype=np.float32)
    allocations = []

    def upload(array):
        pointer = C.c_void_p()
        check(runtime, runtime.cudaMalloc(C.byref(pointer), array.nbytes))
        allocations.append(pointer)
        check(runtime, runtime.cudaMemcpy(pointer, array.ctypes.data, array.nbytes, 1))
        return pointer

    try:
        weights_gpu = upload(raw)
        bank_gpu = upload(packed["bank"])
        x_gpu = upload(x)
        output_gpu = upload(output)
        ids_gpu = upload(ids)
        gathered_gpu = upload(gathered)
        check(runtime, kernels.spin8_g128_cuda_matmul_f32(weights_gpu, bank_gpu, x_gpu, output_gpu,
                                                         rows, width, 3, width + 5, rows + 3, None))
        check(runtime, kernels.spin8_g128_cuda_get_rows_f32(weights_gpu, bank_gpu, ids_gpu, gathered_gpu,
                                                           rows, width, len(ids), None))
        check(runtime, runtime.cudaDeviceSynchronize())
        check(runtime, runtime.cudaMemcpy(output.ctypes.data, output_gpu, output.nbytes, 2))
        check(runtime, runtime.cudaMemcpy(gathered.ctypes.data, gathered_gpu, gathered.nbytes, 2))
        expected = x[:, :width] @ canonical.T
        np.testing.assert_allclose(output[:, :rows], expected, rtol=5e-5, atol=5e-5)
        np.testing.assert_array_equal(output[:, rows:], np.full((3, 3), 123456.0, dtype=np.float32))
        np.testing.assert_allclose(gathered[:4], canonical[ids[:4]], rtol=3e-5, atol=3e-6)
        assert np.isnan(gathered[4:]).all()
        return {"shape": list(packed["shape"]), "tokens": 3,
                "matmul_max_abs": float(np.max(np.abs(output[:, :rows] - expected))),
                "get_rows_max_abs": float(np.max(np.abs(gathered[:4] - canonical[ids[:4]]))),
                "padded_input_output_strides_checked": True, "invalid_ids_produce_nan_without_reading_weights": True}
    finally:
        for allocation in reversed(allocations):
            check(runtime, runtime.cudaFree(allocation))


def run(runtime, kernels):
    p = C.c_void_p
    kernels.spin8_g128_cuda_matmul_f32.argtypes = [p, p, p, p, *([C.c_int64] * 5), p]
    kernels.spin8_g128_cuda_get_rows_f32.argtypes = [p, p, p, p, *([C.c_int64] * 3), p]
    kernels.spin8_g128_cuda_matmul_f32.restype = C.c_int
    kernels.spin8_g128_cuda_get_rows_f32.restype = C.c_int
    rng = np.random.default_rng(20261120)
    bank = ref.make_bank()
    weight = rng.normal(scale=0.1, size=(9, 384))
    weight[1] = 0
    weight[2, 17] = 3
    cases = {"encoded": run_case(runtime, kernels, ref.encode(weight, bank), 101)}
    for mode, count, sign in (("all_ternary", 65498, 1), ("all_uniform", 65536, -1)):
        groups = (count + 15) // 16
        words = np.zeros(groups * 16, dtype=np.uint16)
        words[:count] = np.arange(count, dtype=np.uint16)
        packed = {"words": words.reshape(groups, 16), "scales": np.full(groups, sign, dtype=np.float16),
                  "bank": bank, "shape": (groups, 128), "group_size": 128,
                  "hadamard": True, "identity_fallback": True, "uniform_fallback": True}
        cases[mode] = run_case(runtime, kernels, packed, 102)
    return {"scope": "fused CUDA codec operators only; no language-model quality or throughput claim",
            "cuda_executed": True, "bank_storage": "float64", "kernel_arithmetic": "float32",
            "global_dense_weight_buffer": False, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cudart", default="libcudart.so")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="native")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--report", type=Path, default=Path("cuda-codec-selfcheck.json"))
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()
    report = None
    try:
        runtime = C.CDLL(args.cudart)
        configure_runtime(runtime)
        count = C.c_int()
        status = runtime.cudaGetDeviceCount(C.byref(count))
        if status or count.value < 1:
            report = {"cuda_executed": False, "status": "skipped_no_cuda_device",
                      "device_count": count.value,
                      "runtime_message": runtime.cudaGetErrorString(status).decode() if status else "no CUDA device"}
    except OSError as error:
        report = {"cuda_executed": False, "status": "skipped_missing_cuda_runtime", "runtime_message": str(error)}
    if report is None:
        with tempfile.TemporaryDirectory(prefix="spin8-cuda-check-") as directory:
            library = args.library
            if library is None:
                library = Path(directory) / "libspin8-cuda.so"
                subprocess.run([args.nvcc, "-std=c++17", "-use_fast_math", "-shared", "-Xcompiler=-fPIC",
                                "-DSPIN8_STANDALONE_CUDA", "-arch=" + args.arch,
                                str(ROOT / "ggml/src/ggml-cuda/spin8.cu"), "-o", str(library)], check=True)
            report = run(runtime, C.CDLL(str(library.resolve())))
            report["device_count"] = count.value
    text = json.dumps(report, indent=2) + "\n"
    args.report.write_text(text)
    print(text, end="")
    if args.require_device and not report["cuda_executed"]:
        raise SystemExit(2)
