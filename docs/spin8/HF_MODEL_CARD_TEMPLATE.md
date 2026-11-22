---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - spin8
  - experimental
  - text-generation
---
# Qwen3.8-27B - Spin8 G128 GGUF

This is an experimental, text-only Spin8 quantisation of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/@SOURCE_REVISION@), produced from the pinned source weights on a Google Colab A100. The finished file and its execution checks are identified below. This release uses the ordinary quantisation baseline: no mathematical task vector, direction removal, additional answer data or optimiser steps were applied.

**This GGUF requires the supplied patched llama.cpp runtime.** Its Spin8 tensor type is an unregistered extension. Standard upstream llama.cpp, Ollama and LM Studio loaders do not support this representation merely because the file has a `.gguf` suffix. The maintained implementation is [zapabob/llama.cpp-spin8](https://github.com/zapabob/llama.cpp-spin8); the exact runtime payload supplied with this model is the reproducible reference.

## Release identity

| Item | Recorded value |
| --- | --- |
| Hugging Face repository | [`zapabobouj/Qwen3.8-27B-Spin8-GGUF`](https://huggingface.co/zapabobouj/Qwen3.8-27B-Spin8-GGUF) |
| Model file | [`@MODEL_FILE@`](@MODEL_FILE@) |
| Complete file size | @MODEL_BYTES@ bytes (@MODEL_GIB@ GiB) |
| SHA-256 | `@MODEL_SHA256@` |
| Conversion completed | @CONVERSION_COMPLETED@ |
| Model verification completed | @VERIFICATION_COMPLETED@ |
| Source revision | `@SOURCE_REVISION@` |
| Source content fingerprint | `@SOURCE_FINGERPRINT@` |
| Conversion configuration fingerprint | `@CONFIG_FINGERPRINT@` |
| llama.cpp upstream commit | `@BASE_COMMIT@` |
| Runtime patch SHA-256 | `@PATCH_SHA256@` |
| Verified conversion device | @GPU_NAME@, @GPU_GIB@ GiB reported device memory |
| Conversion software | PyTorch @TORCH_VERSION@; CUDA build @CUDA_VERSION@ |

These values are taken from completed conversion, build and verification records. The card renderer also checks the final GGUF's byte length, header and SHA-256. The [provenance record](provenance.json), [conversion report](conversion.json), [verification report](verification.json) and [build report](evidence/build-report.json) retain the detailed identities.

## What was converted

The source was downloaded at revision `@SOURCE_REVISION@`. Its **@SHARD_COUNT@ Safetensors shards** contain **@SOURCE_FILE_BYTES@ file bytes**, including headers, and **@SOURCE_PAYLOAD_BYTES@ tensor-payload bytes**. The completed [source manifest](evidence/source.json) records the downloaded inventory and hashes.

The exported model contains the text component. Vision tensors and the multi-token prediction component are excluded; this release does not provide image input or MTP inference. The original source totals therefore should not be used as a like-for-like denominator for the text-only GGUF's compression ratio.

The pipeline maps the source into the pinned converter's canonical Qwen35 tensor layout before quantisation. Supported aligned attention, GDN, feed-forward, output and embedding matrices use Spin8; other text tensors retain their applicable floating storage. The converter streams tensors and tiles without staging a complete floating-point GGUF.

| Tensor accounting | Count or size |
| --- | --- |
| Canonical text tensors before companion banks | @CANONICAL_TENSORS@ |
| Matrices encoded as Spin8 G128 | @QUANTISED_TENSORS@ |
| Weights within those compressed matrices | @QUANTISED_WEIGHTS@ |
| Packed Spin8 matrix payload | @PACKED_BYTES@ bytes |
| Companion rotation banks | @BANK_BYTES@ bytes |
| Total GGUF tensor entries, including banks | @GGUF_TENSORS@ |

## Representation and settings

Spin8 G128 stores each group of 128 weights as sixteen 16-bit code words and one 16-bit scale: **34 bytes per group, or 2.125 bits per quantised weight**. The rate is a payload rate. Each compressed matrix also has a 4,608-byte F64 rotation bank; floating tensors, the tokenizer, metadata and alignment add to the complete file size.

The codec combines a normalised Hadamard transform with eight-dimensional rotation frames and ternary codes. It also includes identity and uniform-code fallbacks, so the complete representation is not exclusively ternary. The Spin(8) triality construction supplies a reproducible fixed rotation bank. It does not run three language models, and the bank was not trained on mathematical answers or a language-model objective.

| Setting | Value |
| --- | --- |
| Bank construction | Fixed `triality` bank |
| Bank seed | `81237` |
| Encoder | PyTorch on CUDA |
| Search arithmetic | `float32` |
| Candidate scale steps | `17` |
| Groups per encoder chunk | `8192` |
| Outer tile size | `1048576` elements |
| Embedding quantisation | Enabled |
| Mathematical weight interpolation | Disabled; effective alpha `0` |
| Direction projection | None |
| Additional answer data | None |
| Optimiser steps | `0` |

The ABI uses private GGML type **44**, file type **43**, codec version **1**, and canonical architecture metadata `qwen35`. These numeric assignments are specific to the pinned patched runtime and are not upstream registrations. Use the accompanying patch, GGUF Python definitions and native kernels together.

## Validation

### Synthetic numerical checks

The bootstrap passed its CPU codec, CUDA codec and tiny-model graph checks. The graph comparisons use the same reconstructed weights on both sides: native Spin8 CUDA execution is compared with a decoded floating CPU control. They test the codec's integration and arithmetic; they do not measure the loss of accuracy from quantising the original model.

| Check | Logit elements | Maximum absolute error | RMSE | Tolerance violations |
| --- | --- | --- | --- | --- |
@PARITY_ROWS@

The tolerance remains `atol = 1e-4`, `rtol = 1e-4`, applied as `abs(actual - reference) <= atol + rtol * abs(reference)`. The checks require observed `MUL_MAT_SPIN8` output buffers on CUDA. This placement evidence is narrower than a kernel-profiler trace.

Both comparison processes use F32 K/V caches. Their ordinary F32 matrix multiplications are sent to the explicit F32 cuBLAS path with `GGML_CUDA_SPIN8_VALIDATE_F32=1` and `NVIDIA_TF32_OVERRIDE=0`. This avoids the custom TF32 MMF path and isolates codec/runtime parity from reduced-precision cache and matrix arithmetic. The compressed Spin8 kernels are unchanged. The settings are scoped to the bootstrap's two parity child processes; the normal server checks and quantiser do not inherit those overrides from the bootstrap.

The recorded comparison profile is:

```json
@PRECISION_PROFILE@
```

See the [ordinary parity report](evidence/native-parity.json) and [projected parity report](evidence/projected-native-parity.json). The projected fixture tests a runtime feature; **no direction projection was applied to this released model**.

### Checks on this completed model

The actual GGUF passed loading, finite-logit validation and observed native Spin8 CUDA placement. It produced **@LOGIT_ELEMENTS@ finite logit values** in the short native execution check, with F16 K/V caches. The API smoke test passed both `/completion` and `/v1/completions` through the normal server configuration. Details are in the [native execution record](evidence/native-execution.json) and [verification report](verification.json).

These are execution checks with short prompts; the native check requested a @NATIVE_CONTEXT@-token context. They do not establish faithful reproduction of the original HF model, long-context stability, mathematical reasoning quality, refusal behaviour or a speed advantage. The example server context below is a user setting, not an evaluated context-length claim.

### Optional mathematical benchmark

@BENCHMARK_SECTION@

No improvement in mathematical ability or refusal behaviour is claimed. A measured result, when present, applies to the exact file, runtime, prompt protocol and sample recorded alongside it.

## Download and run

The following commands are intended for a Linux CUDA environment such as a Colab A100 runtime. Keep Colab's installed CUDA-enabled PyTorch. The bootstrap probes the existing environment, builds the pinned native runtime, and runs its mandatory checks. Public model download does not require a write token.

### 1. Download one immutable release snapshot

```bash
python -m pip install --upgrade huggingface_hub
python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
repo = "zapabobouj/Qwen3.8-27B-Spin8-GGUF"
revision = HfApi().model_info(repo).sha
snapshot_download(repo_id=repo, revision=revision,
                  local_dir="spin8-release", max_workers=2)
print("Downloaded release commit:", revision)
PY
cd spin8-release
```

The resolved commit is printed for repeat downloads. Keep that value with your experiment records. See Hugging Face's [snapshot download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/download).

Check the model file before loading it:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
path = Path("@MODEL_FILE@")
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
assert path.stat().st_size == @MODEL_BYTES_RAW@, "GGUF byte length differs"
assert digest.hexdigest() == "@MODEL_SHA256@", "GGUF SHA-256 differs"
print("GGUF size and SHA-256 verified")
PY
```

### 2. Build the matching runtime

Run from the downloaded `spin8-release` directory:

```bash
python runtime/colab_app/bootstrap.py prepare \
  --work-dir /content/spin8-runtime \
  --payload-dir runtime \
  --device cuda --jobs 4
```

Use a fresh work directory if it already contains a different source revision or payload. The bootstrap verifies the bundled manifest and applies `runtime/spin8-llama.patch` to upstream commit `@BASE_COMMIT@`. Its checks include real CUDA execution. A successful compile alone is insufficient.

### 3. Start the API server

```bash
env -u GGML_CUDA_SPIN8_VALIDATE_F32 \
    -u NVIDIA_TF32_OVERRIDE \
    -u GGML_CUDA_CUBLAS_COMPUTE_TYPE \
  /content/spin8-runtime/llama.cpp/build-spin8-v2-cuda/bin/llama-server \
    -m @MODEL_FILE@ -ngl 99 -c 2048 -b 32 -ub 32 -np 1 \
    --host 127.0.0.1 --port 8080 --no-ui --flash-attn off
```

In another terminal:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write one sentence about prime numbers.","n_predict":64,"temperature":0,"seed":42}'
```

The server exposes its native completion endpoint and OpenAI-compatible completion API. This example uses a plain completion prompt. Chat-template behaviour, longer generations and application-specific prompts require their own checks. An embedded web UI is disabled in this build; clients can connect to the API.

For a CPU build, use `--device cpu`, then the `build-spin8-v2-cpu/bin/llama-server` executable with `-ngl 0`. The recorded full-model verification for this release used CUDA; the CPU instruction is a supported build route, not a reported full-model performance measurement.

## Reproducibility and limits

Use the supplied runtime payload rather than assuming that a future `main` branch or an unrelated GGUF loader implements this ABI. The [public runtime repository](https://github.com/zapabob/llama.cpp-spin8) contains the source, implementation notes and Colab workflow. The complete conversion identity records dependency versions, application hashes, source hashes, bank settings and native binary identities.

This is a lossy experimental quantisation. Smaller storage does not establish better answers or faster inference. No comparison with Bonsai, standard Q4/IQ formats or the original BF16 source is implied by the execution checks. GPU memory in the identity table is device capacity, not a measured peak allocation. No full-model throughput, perplexity or peak-memory figure is supplied unless a matching completed benchmark explicitly records it.

## Licence, attribution and modifications

The source model is by the **Qwen team**. Its pinned model card declares Apache-2.0. The source licence is retained verbatim in the following file or files:

@LICENCE_LINKS@

The accompanying [upstream model card](upstream/source/README.md) preserves source attribution and usage information. Any supplied upstream `NOTICE` is retained alongside it. Consult these original files for the terms governing the weights and their derivatives.

This derivative changes the source weight representation to Spin8 G128, exports the text component, and omits vision and MTP components. It applies no mathematical task-vector edit or refusal-direction edit. Those are the modifications represented by this release.

The native runtime derives from **llama.cpp/ggml and its contributors**, with its MIT licence retained at [runtime/licenses/LICENSE-llama.cpp](runtime/licenses/LICENSE-llama.cpp). The Spin8 extension is maintained in [zapabob/llama.cpp-spin8](https://github.com/zapabob/llama.cpp-spin8). AI assistance was used in the implementation, investigation and documentation. This is an independent derivative, not an official Qwen quantisation.
