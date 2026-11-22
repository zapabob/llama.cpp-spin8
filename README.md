English | [日本語](README.ja.md)

# llama.cpp-spin8

**An experimental llama.cpp extension for Spin8 G128 quantisation.** It stores compressed weights in GGUF and executes them through dedicated CPU/CUDA operations and the Qwen35 graph. The repository includes quantisation tools, direct weight editing, projection onto the subspace orthogonal to a supplied direction, and a Colab workflow for downloading, converting and evaluating Hugging Face models.

The Spin8 implementation is maintained on `main`; imported upstream tags and branches are historical references, not Spin8 releases.

The upstream base is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit [`7fce5439fe451cd48340b112ec37d2befe02c776`](https://github.com/ggml-org/llama.cpp/commit/7fce5439fe451cd48340b112ec37d2befe02c776). The original introduction is preserved in the [upstream README](docs/spin8/UPSTREAM_README.md).

## Current status

As recorded on **23 November 2026 (Japan Standard Time)**:

| Area | Status |
| --- | --- |
| Spin8 CPU/CUDA operations, loader and Qwen35 graph integration | Implemented |
| CPU inference on small synthetic models, including projection | Verified |
| HF download, conversion, resumption and evaluation tools | Implemented, with small-model checks |
| Native CPU/CUDA codec, graph parity, projected graph parity and generation API checks | Passed on a Colab A100-SXM4-40GB |
| Qwen3.8-27B source download | All 18 shards downloaded and verified |
| Full 27B Spin8 conversion | Running |
| Completed 27B GGUF inference, mathematics benchmarks and HF publication | Not yet completed |

The A100 run used PyTorch `2.14.1+cu130` and CUDA `13.0`. With the explicit FP32 comparison profile described below, the maximum absolute logit errors were `3.4570693969726562e-6` for the ordinary fixture and `3.427267074584961e-6` for the projected fixture. Both passed the unchanged `atol=rtol=1e-4` limits and observed `MUL_MAT_SPIN8` execution on CUDA.

[STATUS.md](docs/spin8/STATUS.md) records the validation scope and source artefact hashes. These comparisons use synthetic models and CPU references decoded from the same Spin8 weights; they do not measure quality against the original BF16 model. The repository contains no pretrained 27B weights. No improvement in mathematical ability or refusal rate has been measured, and full-model throughput has not yet been reported.

GitHub Actions was disabled when this repository was created to prevent imported upstream release workflows from running. The recorded CPU results come from earlier local validation; they are not evidence of a successful CI run in this repository.

## Format and scope

- Each group of 128 weights occupies 34 bytes: **2.125 bits per weight for the compressed payload**. Per-matrix rotation banks, GGUF metadata and tensors that remain uncompressed add to the final size.
- The codec uses a Hadamard transform and a bank of eight-dimensional rotations. The default triality bank is fixed and reproducible from its seed. The CLI does not train rotations or run three LLMs.
- GGML type **44**, file type **43** and codec version **1** are unregistered extensions specific to this pinned runtime. Standard upstream llama.cpp, Ollama and LM Studio loaders cannot read this representation.
- Conversion currently targets canonical `qwen35` GGUF files. Dedicated CPU/CUDA operations handle compressed matrix multiplication and embedding lookup. Support across all model families has not been established.
- Mathematical donor differences and projection onto the subspace orthogonal to a supplied direction are optional. The tools do not discover a direction automatically; its meaning and its effect on model behaviour require separate evaluation.

See the [Spin8 documentation](tools/spin8/README.md) and [testing guide](tools/spin8/TESTING.md) for the detailed format and execution contract.

## Build for CPU and check a small model

On Linux, install CMake, a C/C++ compiler and Python, then clone the repository:

```sh
git clone https://github.com/zapabob/llama.cpp-spin8.git
cd llama.cpp-spin8
```

Run the following from the repository root. This builds an API server without downloading external Web UI assets.

```sh
cmake -S . -B build-spin8-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_RPC=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-spin8-cpu --target llama-server test-spin8-native -j 4

python tools/spin8/verify_native.py \
  --binary build-spin8-cpu/bin/test-spin8-native \
  --model examples/spin8-colab/payload/fixtures/tiny-qwen35-spin8.gguf \
  --control examples/spin8-colab/payload/fixtures/tiny-qwen35-decoded.gguf \
  --out-dir build-spin8-cpu/spin8-parity \
  --ngl 0 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 --require-spin8-device CPU
```

This comparison checks native execution against an F32 model decoded from the same Spin8 weights. It does not measure quantisation error against the original pretrained model or assess language quality.

## Build for an A100 with CUDA

Install the CUDA Toolkit and a compatible compiler. Use a separate build directory for CUDA; architecture `80` is the A100 setting.

```sh
cmake -S . -B build-spin8-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGGML_NATIVE=OFF -DGGML_RPC=OFF \
  -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-spin8-cuda --target llama-server test-spin8-native -j 4

GGML_CUDA_SPIN8_VALIDATE_F32=1 NVIDIA_TF32_OVERRIDE=0 \
python tools/spin8/verify_native.py \
  --binary build-spin8-cuda/bin/test-spin8-native \
  --model examples/spin8-colab/payload/fixtures/tiny-qwen35-spin8.gguf \
  --control examples/spin8-colab/payload/fixtures/tiny-qwen35-decoded.gguf \
  --out-dir build-spin8-cuda/spin8-parity \
  --ngl 99 --control-ngl 0 --cache-type f32 \
  --atol 1e-4 --rtol 1e-4 \
  --require-spin8-device CUDA --require-spin8-op MUL_MAT_SPIN8
```

This is the profile used for the completed A100 synthetic comparisons. F32 K/V storage gives both executions the same cache precision. The opt-in `GGML_CUDA_SPIN8_VALIDATE_F32` switch sends ordinary F32 matrix multiplication through cuBLAS F32, avoiding the custom MMF path's TF32 instructions; `NVIDIA_TF32_OVERRIDE=0` disables TF32 in cuBLAS. These settings leave Spin8 kernels unchanged and apply only to this comparison process. The Colab bootstrap records the profile and runs its generation API smoke checks with the normal environment.

The placement check detects ordinary CPU fallback; it is neither a throughput benchmark nor a GPU kernel profiler. See the [testing guide](tools/spin8/TESTING.md) for the projected fixture and recorded comparison fields.

## Quantisation and direct weight editing

Use [convert.py](tools/spin8/convert.py) to quantise a floating-point GGUF, and [edit_weights.py](tools/spin8/edit_weights.py) for mathematical weight differences and direction projection. The [Spin8 documentation](tools/spin8/README.md) describes Python dependencies and input requirements. The CUDA encoder requires a CUDA-enabled PyTorch installation.

```sh
python tools/spin8/convert.py quantize model-f16.gguf model-spin8.gguf \
  --include-embedding --encoder torch --device cuda \
  --compute-dtype float32 --chunk-groups 8192 \
  --resume-dir spin8-cache/run-01

build-spin8-cuda/bin/llama-server -m model-spin8.gguf \
  --host 127.0.0.1 --port 8080 -ngl -1
```

Retain the source model identity, conversion settings and provenance of any edited weights. Direction projection is a linear operation on a supplied vector; it does not guarantee refusal removal or improved mathematical ability. This weight-editing pipeline performs no answer-data SFT or optimiser updates.

## Colab and benchmarks

The repository includes the [Colab notebook](examples/spin8-colab/Spin8_A100_Weight_Edit.ipynb) and a [Japanese execution guide](examples/spin8-colab/README_JA.md).

[Open in Google Colab](https://colab.research.google.com/github/zapabob/llama.cpp-spin8/blob/main/examples/spin8-colab/Spin8_A100_Weight_Edit.ipynb)

The notebook is available from this public repository. You can also download it and upload it to Colab.

The notebook embeds a verified source payload and applies its bundled patch to the pinned upstream commit. It does not automatically follow future changes to this repository's `main` branch.

It checks the allocated GPU before substantial work and stops if Colab supplies a device other than an A100. It preserves the installed PyTorch/CUDA environment and streams stage output into the notebook. The encoder defaults are `chunk_groups=8192`, `tile_elements=1048576`, `scale_steps=17` and seed `81237`.

The defaults perform ordinary Spin8 conversion and evaluate 64 GSM8K questions. Set `MATH_DELTA=True` to enable the mathematical weight difference. To enable direction projection, set `DIRECTION_PATH` to an existing direction file appropriate to the model. Set `BENCHMARK_FULL=True` to evaluate all 1,319 questions. Comparing against the original F16 model requires additional disk space; if a reference model cannot be prepared, improvement is recorded as unmeasured. The workflow does not include a refusal-rate benchmark.

## Licence and attribution

[MIT Licence](LICENSE). The upstream copyright notices and licence are retained. The underlying llama.cpp/ggml project is the work of The ggml authors. AI assistance was used in the design, implementation and validation of the Spin8 extension. Model weights remain subject to their respective licences.
