# Native Spin8 execution tests

These tests compare native graph execution with an F32 model decoded from the same Spin8 payload. They do not measure error relative to the original BF16 model, language quality, or production throughput. A tiny fixture passing these tests is not a completed 27B model.

## Build the executable

From the repository root, enable `LLAMA_BUILD_TESTS` and `LLAMA_BUILD_COMMON` in the chosen CPU or CUDA build, then build the test target:

```sh
cmake --build build-spin8 --target test-spin8-native -j 4
```

`test-spin8-native` uses the public libllama API and explicit token IDs. It does not need a tokeniser to run a fixture with `tokenizer.ggml.model=none`. Its defaults are prompt `[1,4,7,9,13]` followed by single-token, teacher-forced steps `[17,21,25]`. It writes all vocabulary logits at all eight input positions, preserving recurrent and attention state between calls.

## Compare against decoded weights

Use the converter's `--decoded-control` output as the reference. An original unquantised checkpoint is a different accuracy experiment.

```sh
python tools/spin8/verify_native.py \
  --binary build-spin8/bin/test-spin8-native \
  --model /path/to/tiny-qwen35-spin8.gguf \
  --control /path/to/tiny-qwen35-decoded.gguf \
  --out-dir /path/to/parity-results \
  --ngl 0 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 --require-spin8-device CPU
```

For a CUDA build, use the following FP32 validation profile to compare the GPU implementation against the decoded CPU reference. The environment assignments apply only to this command and its child processes:

```sh
GGML_CUDA_SPIN8_VALIDATE_F32=1 NVIDIA_TF32_OVERRIDE=0 \
python tools/spin8/verify_native.py \
  --binary build-spin8-cuda/bin/test-spin8-native \
  --model /path/to/tiny-qwen35-spin8.gguf \
  --control /path/to/tiny-qwen35-decoded.gguf \
  --out-dir /path/to/cuda-parity-results \
  --ngl 99 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 \
  --require-spin8-device CUDA --require-spin8-op MUL_MAT_SPIN8
```

For the projected fixture, use `tiny-qwen35-projected-spin8.gguf` as `--model`, `tiny-qwen35-projected-control.gguf` as `--control`, and a separate output directory. Keep the same environment assignments, cache type and tolerances.

`--control-ngl` defaults to the same setting as `--ngl`. Setting it to `0` keeps the F32 control on CPU and avoids interpreting a lower-precision GPU F32 multiplication path as the exact reference. The numerical tolerances remain `atol=1e-4` and `rtol=1e-4`; the validation profile does not relax them.

`--cache-type f32` selects F32 storage for both K and V in both executions. The harness defaults to `f16` when this option is omitted. The verifier checks that the recorded cache types agree with each other and with the requested setting.

The CUDA MMF path can use explicit TF32 instructions for ordinary F32 matrix multiplication. `NVIDIA_TF32_OVERRIDE=0` alone does not disable those custom instructions. With `GGML_CUDA_SPIN8_VALIDATE_F32=1`, ordinary `MUL_MAT` operations whose weights, activations and output are all F32 use the existing cuBLAS F32 implementation instead; the NVIDIA setting disables TF32 for that cuBLAS execution. This switch does not change Spin8 kernels, does not cover `MUL_MAT_ID` or separately fused operations, and is off by default. The native observation callback supplies the graph boundaries used by this test.

The Colab bootstrap applies this profile only to its normal and projected parity child processes. Its codec checks and server API smoke test retain their normal execution settings. Do not treat the parity profile as a measurement of default server throughput.

The placement requirement applies to `MUL_MAT_SPIN8` by default. `--require-spin8-op GET_ROWS_SPIN8` selects embedding gathers, while `all` requires every observed Spin8 output to match the requested device. llama.cpp can intentionally keep token embeddings on CPU while dense projections run on GPU. The report always lists all observed Spin8 placements, including those outside the selected requirement.

Placement comes from graph-node output buffers observed after execution. This detects ordinary CPU fallback, but it is not a GPU kernel profiler. The observation callback may inhibit fusion or reduce throughput; these runs are not speed benchmarks.

`--mmap` and `--no-mmap` force the model loading mode for both models. Without either option, the runtime chooses its default mode.

Each result includes:

- `.f32`: complete row-major float logits, with byte order and shape in JSON.
- `.json`: tokens, positions, finite checks, top tokens, model dimensions, devices, output-buffer placements, and `cache_type_k`/`cache_type_v`.
- `.log`: native runtime messages.
- `parity.json`: tolerances, actual errors, top-token agreement, cache types, and SHA-256 hashes of both GGUFs, the executable, logits, and loaded libllama/libggml shared libraries on Linux. Its `precision_environment` records `GGML_CUDA_SPIN8_VALIDATE_F32`, `NVIDIA_TF32_OVERRIDE` and `GGML_CUDA_CUBLAS_COMPUTE_TYPE`; unset values are `null`.

The bootstrap also writes `build-report.json.parity_precision_profile`, including the cache types, tolerances, CPU reference, precision environment and the fact that the server smoke test does not use the parity environment.

## Recorded A100 validation: 23 November 2026

The complete native bootstrap passed on an **NVIDIA A100-SXM4-40GB** in Colab with **Torch 2.14.1+cu130 and CUDA 13.0**. The installed Torch/CUDA environment was preserved.

| Check | Recorded result |
|---|---|
| CPU codec | PASS |
| CUDA codec | PASS with actual device execution |
| Tiny native Spin8 graph against decoded CPU weights | PASS; maximum absolute error `3.4570693969726562e-6` |
| Tiny projected Spin8 graph against projected CPU control | PASS; maximum absolute error `3.427267074584961e-6` |
| Required Spin8 placement | `MUL_MAT_SPIN8` observed on CUDA in both parity checks |
| Generation APIs | PASS with the normal server settings |

Both parity checks used the FP32 profile above and the unchanged `atol=rtol=1e-4` limits, with no tolerance violations. The reports are `reports/build-report.json`, `reports/native-parity/parity.json`, `reports/projected-native-parity/parity.json` and the codec/server reports in the same work directory. This verifies execution against the same decoded codec weights, not accuracy relative to the original BF16 model.

At this revision, the full 27B conversion is still running. Verification of a completed 27B GGUF, its quality benchmark and its HF upload are not yet complete. No mathematical-ability improvement or reduction in refusal behaviour has been established.

## Malformed loader inputs

```sh
python tools/spin8/test_loader.py \
  --binary build-spin8/bin/test-spin8-native \
  --fixture /path/to/tiny-qwen35-spin8.gguf \
  --out-dir /path/to/loader-results
```

This script accepts fixtures up to 32 MiB and writes malformed copies. It checks missing, mistyped, and unsupported codec versions; missing or malformed banks; nonorthogonal banks; invalid scales; reserved ternary codes; and truncated payloads. Each case runs with and without mmap. Success requires a clean model-load error, with no signal or abort. The source fixture is hashed before and after the test.

## CTest integration

Provide two existing tiny fixture paths when configuring:

```sh
cmake -S . -B build-spin8 \
  -DSPIN8_TEST_MODEL=/path/to/tiny-qwen35-spin8.gguf \
  -DSPIN8_TEST_CONTROL=/path/to/tiny-qwen35-decoded.gguf
cmake --build build-spin8 --target test-spin8-native -j 4
ctest --test-dir build-spin8 -R '^test-spin8-native-(parity|loader)$' --output-on-failure
```

The CTest entries use CPU execution and the selected Python interpreter. The loader test also needs NumPy and the repository's GGUF Python module. No weights or datasets are downloaded by these tests.

For MTP skip coverage, use a fixture with declared NextN weights and their Spin8 banks. The harness keeps `load_mtp=false`, matching ordinary trunk inference. Passing this case verifies unused-bank accounting; it does not establish active MTP graph correctness.

## Optional direction removal across quantisation

`edit_weights.py --remove-direction direction.npy` edits floating weights before quantisation. For a finite nonzero model-specific hidden direction `d`, it uses `P = I - d dᵀ/(dᵀd)`. The native runtime applies the same outer projection so rounding and Spin8 encoding cannot silently turn the effective linear map back into an unprojected map:

- Residual writers (`blk.N.attn_output.weight`, `blk.N.ffn_down.weight`, `blk.N.ssm_out.weight`) use `P Q(P W)`.
- The token embedding table uses `Q(E P) P`. Token lookup projects the hidden vector after inverse Hadamard and embedding LoRA. If the table is tied to the output head, the head projects its input hidden vector; it does not project the vocabulary axis.
- The left projection follows the matrix scale and LoRA contribution; right projection precedes the Hadamard transform and LoRA input.

The GGUF contract is `spin8.projection.version` as UINT32 `1`, `spin8.projection.left_weights` and `.right_weights` as STRING arrays, and an auxiliary F32 vector `spin8.projection.direction` of length `hidden_size`. The right array currently permits only `token_embd.weight`. Known skipped NextN residual writers can remain named, and bind only when those weights are loaded. A shared MTP embedding uses the same projection; a separate MTP embedding table is outside this version's right-name contract.

The loader validates names, axes, version types, and finite nonzero direction data. It normalises the stored vector in FP64 into small F32 device buffers without modifying the file or an mmap. Arithmetic remains floating point: residual direction components can be at numerical rounding scale. A `--no-alloc` harness run constructs the same projection graph shapes with dummy buffers for memory planning, without reading payloads or executing inference.

```sh
python tools/spin8/test_projection.py \
  --binary build-spin8/bin/test-spin8-native \
  --float-fixture /path/to/tiny-qwen35-f32.gguf \
  --out-dir /path/to/new-empty-projection-results
```

This test creates a synthetic direction, calls the actual floating-weight editor and Spin8 converter, then compares native logits with a decoded F32 control whose final effective projected matrices are independently materialised. It covers both-sided metadata, left-only and right-only lists, a tied output head, a very small nonzero direction, the memory-fit graph, and malformed projection inputs with and without mmap. Numerical tolerances remain `atol=rtol=1e-4`. The test checks algebra and integration only: its synthetic direction is not evidence of a refusal feature, and neither projection nor passing logits parity establishes a behavioural or mathematical-ability improvement.
