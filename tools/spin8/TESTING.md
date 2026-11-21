# Native Spin8 execution tests

These tests compare native graph execution with an F32 model decoded from the same Spin8 payload. They do not measure error relative to the original BF16 model, language quality, or production throughput. A tiny fixture passing these tests is not a completed 27B model.

## Build the executable

From the repository root, enable `LLAMA_BUILD_TESTS` and `LLAMA_BUILD_COMMON` in the chosen CPU or CUDA build, then build the test target:

```sh
cmake --build build-spin8 --target test-spin8-native -j 4
```

`test-spin8-native` uses the public libllama API and explicit token IDs. It does not need a tokenizer to run a fixture with `tokenizer.ggml.model=none`. Its defaults are prompt `[1,4,7,9,13]` followed by single-token, teacher-forced steps `[17,21,25]`. It writes all vocabulary logits at all eight input positions, preserving recurrent and attention state between calls.

## Compare against decoded weights

Use the converter's `--decoded-control` output as the reference. An original unquantized checkpoint is a different accuracy experiment.

```sh
python tools/spin8/verify_native.py \
  --binary build-spin8/bin/test-spin8-native \
  --model /path/to/tiny-qwen35-spin8.gguf \
  --control /path/to/tiny-qwen35-decoded.gguf \
  --out-dir /path/to/parity-results \
  --ngl 0 --control-ngl 0 --require-spin8-device CPU
```

For a CUDA build, compare the GPU implementation against the decoded CPU reference:

```sh
python tools/spin8/verify_native.py \
  --binary build-spin8-cuda/bin/test-spin8-native \
  --model /path/to/tiny-qwen35-spin8.gguf \
  --control /path/to/tiny-qwen35-decoded.gguf \
  --out-dir /path/to/cuda-parity-results \
  --ngl -1 --control-ngl 0 \
  --require-spin8-device CUDA --require-spin8-op MUL_MAT_SPIN8
```

`--control-ngl` defaults to the same setting as `--ngl`. Setting it to `0` keeps the F32 control on CPU and avoids interpreting a lower-precision GPU F32 multiplication path as the exact reference. The numeric tolerances remain `atol=1e-4` and `rtol=1e-4` unless explicitly changed.

The placement requirement applies to `MUL_MAT_SPIN8` by default. `--require-spin8-op GET_ROWS_SPIN8` selects embedding gathers, while `all` requires every observed Spin8 output to match the requested device. llama.cpp can intentionally keep token embeddings on CPU while dense projections run on GPU. The report always lists all observed Spin8 placements, including those outside the selected requirement.

Placement comes from graph-node output buffers observed after execution. This detects ordinary CPU fallback, but it is not a GPU kernel profiler. The observation callback may inhibit fusion or reduce throughput; these runs are not speed benchmarks.

`--mmap` and `--no-mmap` force the model loading mode for both models. Without either option, the runtime chooses its default mode.

Each result includes:

- `.f32`: complete row-major float logits, with byte order and shape in JSON.
- `.json`: tokens, positions, finite checks, top tokens, model dimensions, devices, and output-buffer placements.
- `.log`: native runtime messages.
- `parity.json`: tolerances, actual errors, top-token agreement, and SHA-256 hashes of both GGUFs, the executable, logits, and loaded libllama/libggml shared libraries on Linux.

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

## Optional direction removal across quantization

`edit_weights.py --remove-direction direction.npy` edits floating weights before quantization. For a finite nonzero model-specific hidden direction `d`, it uses `P = I - d dᵀ/(dᵀd)`. The native runtime applies the same outer projection so rounding and Spin8 encoding cannot silently turn the effective linear map back into an unprojected map:

- Residual writers (`blk.N.attn_output.weight`, `blk.N.ffn_down.weight`, `blk.N.ssm_out.weight`) use `P Q(P W)`.
- The token embedding table uses `Q(E P) P`. Token lookup projects the hidden vector after inverse Hadamard and embedding LoRA. If the table is tied to the output head, the head projects its input hidden vector; it does not project the vocabulary axis.
- The left projection follows the matrix scale and LoRA contribution; right projection precedes the Hadamard transform and LoRA input.

The GGUF contract is `spin8.projection.version` as UINT32 `1`, `spin8.projection.left_weights` and `.right_weights` as STRING arrays, and an auxiliary F32 vector `spin8.projection.direction` of length `hidden_size`. The right array currently permits only `token_embd.weight`. Known skipped NextN residual writers can remain named, and bind only when those weights are loaded. A shared MTP embedding uses the same projection; a separate MTP embedding table is outside this version's right-name contract.

The loader validates names, axes, version types, and finite nonzero direction data. It normalizes the stored vector in FP64 into small F32 device buffers without modifying the file or an mmap. Arithmetic remains floating point: residual direction components can be at numerical rounding scale. A `--no-alloc` harness run constructs the same projection graph shapes with dummy buffers for memory planning, without reading payloads or executing inference.

```sh
python tools/spin8/test_projection.py \
  --binary build-spin8/bin/test-spin8-native \
  --float-fixture /path/to/tiny-qwen35-f32.gguf \
  --out-dir /path/to/new-empty-projection-results
```

This test creates a synthetic direction, calls the actual floating-weight editor and Spin8 converter, then compares native logits with a decoded F32 control whose final effective projected matrices are independently materialized. It covers both-sided metadata, left-only and right-only lists, a tied output head, a very small nonzero direction, the memory-fit graph, and malformed projection inputs with and without mmap. Numeric tolerances remain `atol=rtol=1e-4`. The test checks algebra and integration only: its synthetic direction is not evidence of a refusal feature, and neither projection nor passing logits parity establishes a behavioral or mathematical-ability improvement.
