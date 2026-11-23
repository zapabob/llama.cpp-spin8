# Native Spin8 G128 prototype

This private llama.cpp extension stores compressed weights in GGUF and executes them through native matrix multiplication and embedding lookup operations. It retains the Qwen35 graph, tokenizer and recurrent state machinery. The synthetic fixture exercises three GDN layers and one full-attention layer; it is not a trained language model.

The upstream base is `7fce5439fe451cd48340b112ec37d2befe02c776`. GGML type **44** and file type **43** are private assignments in this patch, not upstream registrations. Use this patched tree and its `gguf-py`. Stock llama.cpp, Ollama and LM Studio do not acquire support merely because the filename ends in `.gguf`.

## Native contract

| Item | Representation |
| --- | --- |
| Architecture | `general.architecture = qwen35` |
| Codec version | `spin8.codec_version`, typed `UINT32`, value `1` |
| Quantized weight | Canonical tensor name and logical matrix dimensions; private GGML type 44 |
| Group | 128 consecutive input weights within one output row |
| Group bytes | 16 little-endian `uint16` words followed by one signed little-endian FP16 scale: 34 bytes |
| Bank | `<weight_name>.spin8_bank`, F64, NumPy shape `[9,8,8]`, GGML dimensions `[8,8,9]` |
| Matrix operation | `MUL_MAT_SPIN8(weight, activation, bank)` |
| Embedding operation | `GET_ROWS_SPIN8(weight, token_ids, bank)` |

Payload storage is `34 * 8 / 128 = 2.125` bits per weight. Each compressed matrix adds a 4,608-byte bank, tensor metadata and alignment. Unselected tensors retain their original storage.

With a positive scale, words 0 through 59,048 represent nine frames times 6,561 ternary vectors. Words 59,049 through 65,497 represent the identity fallback; higher words are reserved. A negative scale selects uniform Q2 mode: all 16-bit words are legal and their eight symbols decode as `(symbol - 1.5) * abs(scale)`. Signed zeros, nonfinite scales and invalid banks are rejected. Reconstruction applies the normalized Sylvester H128 after decoding the 8D vectors.

`reference_codec.py` is the independent NumPy implementation. The default `triality` bank is constructed reproducibly from a seed. The quantization CLI does not train rotations or optimize a language-model objective. `--bank-kind haar` supplies a baseline. Uniform fallback is part of the codec, so the result is not necessarily entirely ternary.

## Canonical conversion first

Start with an ordinary, unquantized, canonical Qwen35 GGUF:

```sh
python convert_hf_to_gguf.py /path/to/hf-checkpoint \
  --outtype bf16 --outfile /path/to/model-bf16.gguf
```

Pin the HF revision and preserve the conversion log. Check the unquantized architecture before interpreting quantization results. These fixtures do not establish Qwen3.8-27B equivalence to HF. The current Qwen35 graph uses sigmoid for the full-attention query gate and SiLU for the GDN output gate; a checkpoint's `output_gate_type` must be matched to the corresponding operation. Converter class registration alone does not prove that match.

Qwen35 conversion includes concatenations and GDN channel/head permutations. This tool encodes already canonical tensor bytes. It rejects HF-layout reference overlays and partial-row pilots instead of renaming them into a full model.

Existing `prism.hadamard.*` metadata is preserved with exact types. Its activation transform remains before the native Spin8 matmul. The codec reconstructs the stored folded weight; its internal H128 does not replace that graph transform. Removing the metadata or folding the weight again changes the computation.

Inputs must be little-endian single-file Qwen35 GGUFs; merge shards first. F32/F16/BF16 projections are supported. Requantization from other quantized types is rejected. Selection uses canonical attention, FFN, GDN, output and NextN matrix kinds. Norms, convolution kernels and unknown matrices remain untouched even if a dimension is divisible by 128. Embeddings require `--include-embedding`. Preserving and skipping MTP weights does not demonstrate active MTP inference.

## Quantization commands

CPU:

```sh
python tools/spin8/convert.py quantize /path/to/model-bf16.gguf \
  /path/to/model-spin8.gguf --include-embedding --encoder numpy \
  --resume-dir /path/to/checkpoints/run-01
```

An A100 with CUDA-enabled PyTorch:

```sh
python tools/spin8/convert.py quantize /content/model-bf16.gguf \
  /content/drive/MyDrive/model-spin8.gguf \
  --include-embedding --encoder torch --device cuda \
  --compute-dtype float32 --chunk-groups 256 --scale-steps 17 \
  --temp-dir /content \
  --resume-dir /content/drive/MyDrive/spin8-checkpoints/run-01
```

This is an executable CUDA encoder path, not a report of an A100 run. CUDA availability is checked before encoding. `--chunk-groups` bounds the search tile; the whole model is not loaded onto the GPU. NumPy and Torch may choose different codes at numerical ties.

For a first selected-projection experiment:

```sh
python tools/spin8/convert.py quantize /path/to/model-bf16.gguf \
  /path/to/one-projection-spin8.gguf \
  --include '^blk\.0\.ffn_down\.weight$' \
  --decoded-control /path/to/one-projection-decoded.gguf
```

`--include` and `--exclude` accept repeatable regular expressions; they cannot override supported tensor kinds. Inputs are preserved. Existing outputs require `--force`.

Final writers flush a complete temporary file beside the destination, acquire an exclusive `.publish.lock`, recheck overwrite policy and publish with `os.replace`. Native, fixture and reference writers require no hardlinks and can use DriveFS. An interrupted publication can leave a lock; remove it only after confirming that the previous writer has stopped. Atomic rename is the local filesystem contract, not a claim that a cloud provider has finished remote synchronization.

### Resume behavior

`--resume-dir` stores a packed payload, F64 bank and JSON record for each completed tensor. The record is published last. Reuse verifies dimensions, byte counts, SHA-256 hashes, legal scales/codes and SO(8) bank validity. Invalid or incomplete entries are re-encoded individually; unrelated files are preserved.

The immutable `run.json` fingerprint binds source SHA-256, encoder/device/precision, scale steps, chunk size, bank seed/kind, selection, embedding policy, Python/NumPy/Torch versions and codec/converter/GGUF Python file hashes. Changed settings fail rather than mix runs. Use a new directory after changing those settings or code. Output paths and optional decoded/reference artifacts may change without invalidating cached tensors. Use one writer per cache directory. There is no mid-tensor checkpoint: an interrupted tensor is recomputed.

The `.conversion.json` report records cache hits, writes and invalidations. Resuming rehashes the source and cached data; keep the source local when possible because that verification costs I/O.

### Transcode and decode

Add `--reference-output /path/to/canonical-reference.gguf` to quantization to create a standard-type I8/F64 reference container. Its custom architecture is `spin8_qwen35`, and metadata binds canonical layout and source hash. It is a storage/reference artifact.

```sh
python tools/spin8/convert.py transcode /path/to/model-bf16.gguf \
  /path/to/canonical-reference.gguf /path/to/model-native.gguf \
  --decoded-control /path/to/model-decoded.gguf

python tools/spin8/convert.py decode /path/to/model-native.gguf \
  /path/to/model-decoded.gguf
```

Transcoding preserves payloads/banks without re-encoding. Decode expands compressed matrices to F32 and retains other tensors and graph metadata. This control checks execution against the same reconstructed weights. Comparison with original BF16 weights is a separate quality evaluation. Requantizing the decoded file with Q4/IQ adds another approximation and does not preserve this representation.

### Direct weight edits and projection metadata

Canonical float GGUFs may first be processed by `edit_weights.py`. Their `spin8.edit.*` provenance is preserved, including source hashes and the edit recipe. The converter also preserves `spin8.projection.*` and the F32 vector `spin8.projection.direction`; the vector is not quantized.

The projection contract uses version `UINT32=1`, a hidden-dimension vector `d`, and typed string arrays `left_weights` and `right_weights`. For `P = I - dd^T/(d^T d)`, residual writers use left projection and embedding tables use right projection. An optional runtime projection can enforce the selected orthogonality after lossy quantization. Metadata describes an algebraic operation; it does not identify a semantic direction, prove that refusals were removed, or establish a gain in mathematical ability. A supplied direction and any task-vector donor need their own held-out behavioral evaluation. The converter does not create answer data or perform SFT.

The direct edit is `W_edit = W_source + alpha * (W_math_donor - W_math_base)` on selected linear weights. If source and math base are the same checkpoint, this is interpolation. Matching tensor shapes and tokenizer metadata are necessary checks, but do not establish useful task-vector transfer between unrelated training lineages.

```sh
python tools/spin8/edit_weights.py /path/to/source-bf16.gguf \
  /path/to/edited-f16.gguf \
  --math-base /path/to/math-base-bf16.gguf \
  --math-donor /path/to/math-donor-bf16.gguf --alpha 0.25 \
  --remove-direction /path/to/model-specific-direction.npy \
  --outtype f16 --tile-elements 1048576 --temp-dir /content

python tools/spin8/convert.py quantize /path/to/edited-f16.gguf \
  /path/to/edited-spin8.gguf --include-embedding \
  --encoder torch --device cuda --chunk-groups 256 \
  --resume-dir /path/to/checkpoints/edited-run-01
```

Either the task-vector pair or `--remove-direction` may be omitted. The direction must be supplied for this model in its unfolded residual basis; there is no fabricated default. `--math-include` limits arithmetic by regex. Missing donor tensors fail unless explicitly covered by a `--preserve-missing` full-match regex; such a tensor skips task arithmetic but may still receive the separately requested projection. Already compressed or Hadamard-folded inputs are rejected. F64 edit arithmetic is tiled; edited storage is F16 or F32 and post-rounding projection leakage is reported. That numerical report does not establish mathematical-answer improvement or refusal behavior.

## Native execution

Build the CPU backend and a fixture:

```sh
cmake -S . -B build-spin8 -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-spin8 --target llama-server test-spin8-native -j 4

python tools/spin8/make_fixture.py /tmp/spin8-fixture --tokenizer gpt2-byte
python tools/spin8/convert.py quantize /tmp/spin8-fixture/tiny-qwen35-f32.gguf \
  /tmp/spin8-fixture/tiny-qwen35-spin8.gguf --include-embedding \
  --decoded-control /tmp/spin8-fixture/tiny-qwen35-decoded.gguf

python tools/spin8/verify_native.py \
  --binary build-spin8/bin/test-spin8-native \
  --model /tmp/spin8-fixture/tiny-qwen35-spin8.gguf \
  --control /tmp/spin8-fixture/tiny-qwen35-decoded.gguf \
  --out-dir /tmp/spin8-parity --ngl 0 --control-ngl 0 --require-spin8-device CPU
```

For CUDA, configure a separate build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` on an A100. Run parity with `--ngl -1 --control-ngl 0 --require-spin8-device CUDA --require-spin8-op MUL_MAT_SPIN8`. Requested offload alone is insufficient evidence: the harness records output-buffer placement and rejects CPU placement for the selected operation. Embedding lookup may intentionally remain on CPU. See [TESTING.md](TESTING.md) for detailed tests and evidence interpretation.

The patched server can load the native model directly:

```sh
build-spin8/bin/llama-server -m /path/to/model-spin8.gguf \
  --host 127.0.0.1 --port 8080 -ngl 0
```

A frontend connected to this server's compatible HTTP API can use its runtime. That does not add loading support to a frontend's independent local inference engine.

## Evidence and resource scope

Local CPU checks cover synthetic-model logits for prefill and teacher-forced decode, compressed embedding lookup, default skipping of NextN weights/banks, malformed-file rejection and server `/completion`, `/v1/completions` and UI responses. Conversion checks compare every compressed fixture tensor with the independent NumPy decoder and preserve untouched bytes. Resume tests cover reuse, corruption, interrupted publication and settings mismatch. Native reports hash models, executables and loaded libraries.

The earlier local environment had no CUDA GPU. Those CPU results do not establish CUDA compilation/execution, A100 throughput/VRAM peaks, full 27B conversion, perplexity or task quality. A later **A100-SXM4-80GB** run completed the full text-model conversion and passed short native CUDA and generation API checks; see the [completion record](../../docs/spin8/results/qwen38-27b-a100-20261123.json). Its GGUF is 7,167,015,008 bytes, SHA-256 `ae0bc4609f3cee3fad7137da43830b739232b015efb0612769292fbee31c1a60`. The paired 64-question GSM8K job is preparing the F16 baseline, and HF upload has not started. Current native kernels are correctness prototypes; native execution is not a speedup claim. Random fixtures and short execution checks cannot establish useful language behaviour or an advantage over Bonsai, IQ3 or Q4_K_M.

For capacity arithmetic, 27 billion BF16 parameters alone occupy 54 GB (about 50.3 GiB). An all-Spin8 payload would occupy 7.172 GB (about 6.68 GiB), before banks, retained dense tensors and GGUF overhead. Keeping HF weights, a BF16 GGUF, a complete cache and final compressed output would already total about 122.3 GB under these simplified assumptions. Actual tensor counts and retained dense weights determine the real total.

The converter maps source/cached tensors and writes final tensors sequentially. Torch transfers only a tile to the GPU, while packed outputs for the current matrix are allocated on the CPU. BF16 expands to a temporary F32 memory-mapped file for the current matrix. NumPy encoding can allocate F64 data for that matrix. This is bounded by a tensor, not a fixed small RAM budget; page cache, CUDA workspace and tensor geometry affect peaks. Tiled encoding on a 40 GB A100 does not require all 54 GB of BF16 weights to fit in VRAM. The completed full conversion was observed on an 80 GB A100; this does not establish full-run completion on 40 GB or measured peak host/device memory.

For 256 groups, the rotated `[256,16,9,8]` search intermediate alone occupies 1.125 MiB in FP32. Candidates, indexing, reconstruction and PyTorch workspace add to it. FP64 doubles floating-point array storage. Measure a real projection before enlarging the tile; changing `--chunk-groups` changes the cache fingerprint.

An all-F32 decoded 27B control alone would be roughly 108 GB. `--decoded-control` also keeps decoded temporary tensors while writing that output, so disk demand can include both copies. Use controls for tiny fixtures or selected-projection experiments first. Optional reference outputs and persistent caches add packed copies. The conversion report records actual output sizes.

## GGUF Python changes

The byte-tokenizer fixture has a valid empty merge list typed as `ARRAY<STRING>`. Upstream Python writing rejected every empty array, while reading discarded an empty array's element type. This patch permits empty arrays only with an explicit subtype and preserves it on read. Copying tokenizer or projection metadata therefore retains its exact type. The private quantization constant and `(128,34)` geometry are the other GGUF Python extensions.

Run Python conversion/cache checks with existing tiny fixtures:

```sh
python tools/spin8/test_conversion.py /path/to/fixture-directory
python tools/spin8/test_resume.py \
  --fixture /path/to/fixture-directory/tiny-qwen35-f32.gguf \
  --report /path/to/resume-validation.json
```
