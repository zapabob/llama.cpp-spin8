# Spin8 A100 public release notebook

[Spin8_A100_Public_Release.ipynb](Spin8_A100_Public_Release.ipynb) is a lightweight Colab workflow for producing a verified Spin8 GGUF, its model card and a public Hugging Face release. It downloads pinned implementation files and checks their manifest and helper hashes, rather than embedding a large source archive in the notebook.

## Fixed release recipe

| Item | Setting |
|---|---|
| Implementation snapshot | [`3c5b2476ab10b5210f0858addcd4f98ab1394d75`](https://github.com/zapabob/llama.cpp-spin8/tree/3c5b2476ab10b5210f0858addcd4f98ab1394d75) |
| Source model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) |
| Source revision | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| Public upload destination | [`zapabobouj/Qwen3.8-27B-Spin8-GGUF`](https://huggingface.co/zapabobouj/Qwen3.8-27B-Spin8-GGUF) |
| Default work directory | `/content/spin8-public-release` |
| Encoder settings | FP32 search; `chunk_groups=8192`, `scale_steps=17`, `seed=81237`, `tile_elements=1048576` |
| Weight editing | Ordinary Spin8 baseline; no mathematics delta or direction removal |
| Output scope | Text inference; the vision encoder and MTP are excluded |

The resulting GGUF requires the supplied Spin8-patched llama.cpp runtime. Stock GGUF readers are not a supported inference path for this custom format.

## Run and inspect

1. Open the notebook in Colab and select **A100, high memory and the latest recommended runtime**. The workflow checks the actual GPU and executes a small CUDA calculation. An L4 allocation does not satisfy the A100 guard. The installed Torch/CUDA environment is preserved.
2. Run the setup cell to obtain and verify the pinned implementation. The shipped source assignment is `RUN_JOB = False`; set it to `True` when ready to start the model job, then edit that source assignment back to `False` after invoking it.
3. The notebook starts `run_release_job.py` as a detached process. It checks the GPU, builds and tests the native runtime, downloads the pinned HF weights, quantises them, verifies the completed GGUF and prepares the release. The notebook kernel remains available for status checks.
4. Inspect the job's `status.json` and `job.log` under `WORK_DIR/reports/jobs/`. The receipt records the runner PID, process start ticks, stage results and input hashes. A saved `RUNNING` receipt alone is not proof that the process is still alive; use the notebook's status checks when reconnecting.
5. When conversion and verification have passed, the runner invokes `prepare_release.py`. Review the resulting model card and `upload-manifest.json` in `WORK_DIR/release/<config_fingerprint>/`. A `READY` result means the local release is prepared; it has not been uploaded.

The release manifest identifies the exact GGUF, model card, provenance, verification records, required runtime files and upstream licences by size and hash. Release preparation rejects incomplete or mismatched evidence. It does not infer a quality improvement from successful loading or finite logits.

## Publish to Hugging Face

The shipped source assignment is `PUBLIC_UPLOAD = False`. To publish the prepared manifest, provide `HF_TOKEN` through **Colab Secrets**, with permission to write to the fixed destination, and enable the upload cell. After invoking it, edit the source assignment back to `PUBLIC_UPLOAD = False`. Resetting a Python variable does not change saved cell source; a later **Run all** can invoke either action again if its source assignment remains `True`. Keep the token value out of notebook source, configuration and logs; the upload step uses it transiently and does not store it in the release.

The upload helper rechecks every local manifest file, commits that file set to the public repository, and checks the remote sizes and LFS or Git object identities. It records `COMMITTED` before those remote checks and reports `PUBLISHED` only after they pass. The notebook captures the reviewed manifest's SHA-256, checks the publication receipt against that hash, and confirms that the local manifest remains unchanged. Inspect `publication.json` and the resulting HF commit before treating publication as complete.

This notebook publishes the exact manifest recorded in the completed runner's receipt. If a separate workflow regenerates a manifest after adding benchmark results, review the new manifest and use `prepare_release.py check --manifest ...` and `prepare_release.py upload --manifest ...` directly. Preserve the original runner receipt; do not rewrite it or bypass its hash check to admit a different manifest into this notebook.

## Reconnection and persistence

Detaching the job separates it from the notebook kernel. It can continue during a notebook disconnection while its process and Colab VM remain alive. It does not guarantee that Colab will keep the VM running or that the job will finish.

Resume uses the same work directory on the same VM. Completed tensor caches are checked before reuse; an interrupted tensor may need to be encoded again. Do not change the pinned inputs during an active job or start a duplicate runner. This notebook does not mount Drive or provide automatic persistent storage. A replacement or deleted VM can remove `/content`, including downloaded weights, caches and unpublished results.

## Recorded status

The full 27B conversion is COMPLETE and model verification is PASS on **NVIDIA A100-SXM4-80GB**, with Torch `2.14.1+cu130` and CUDA `13.0`. The file `Qwen3.8-27B-Spin8-cg8192-s17-20261123.gguf` contains **7,167,015,008 bytes** (6.675 GiB), SHA-256 `ae0bc4609f3cee3fad7137da43830b739232b015efb0612769292fbee31c1a60`. Conversion completed at `2026-11-23T01:17:31.416927+00:00`; verification completed at `2026-11-23T01:18:41.463919+00:00`.

The native check produced 1,986,560 finite logit values with F16 K/V caches and a requested 128-token context, and observed Spin8 CUDA placement. Both generation APIs passed under normal runtime settings. The [central completion record](../../docs/spin8/results/qwen38-27b-a100-20261123.json) and [status page](../../docs/spin8/STATUS.md) identify the local observation and its scope; no public HF commit is recorded.

The local release manifest is READY. The separate paired 64-question GSM8K job is RUNNING at `baseline-f16`, with no completed comparison result yet. HF upload has not started; publication awaits an HF write credential in Colab. Mathematical quality, refusal behaviour and a speed advantage have not been established by the execution checks.
