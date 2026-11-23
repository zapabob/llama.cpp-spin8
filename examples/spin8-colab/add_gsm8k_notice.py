#!/usr/bin/env python3
"""Add the verified GSM8K notice to a separate READY upload manifest.

Run after the benchmark release job has completed:
  python add_gsm8k_notice.py --manifest /path/to/upload-manifest.json \
    --notice /path/to/GSM8K-NOTICE.txt --release-helper /path/to/prepare_release.py

Only upload-manifest-with-dataset-notice.json is created. The original manifest,
model, card, benchmark evidence and runner receipts are preserved. This command
does not upload. Use the existing release helper's check/upload commands with
the new derived manifest. Identical reruns are accepted; conflicting output is
never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
REPO_ID = "zapabobouj/Qwen3.8-27B-Spin8-GGUF"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
NOTICE_NAME = "benchmarks/LICENSE-GSM8K"
NOTICE_BYTES = 1522
NOTICE_SHA256 = "87f1d2ab3fa4c5779bf1b482ff1469dcbee93c60a731e4f704306a48183c191a"
OUTPUT_NAME = "upload-manifest-with-dataset-notice.json"
HELPER_IDENTITIES = {
    "prepare_release.py": (10928, "50d15915b992d95d0ee4635e99e6e34515000b1baaaecdc1111bd9a81af80566"),
    "render_model_card.py": (22035, "d2f43d587cb7d8c11526328b3b3d87b9a34201360456826cba0101a98753ffc8"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path, size=None, digest=None):
    require(path.is_file() and not path.is_symlink(), "Missing regular file: " + str(path))
    require(path.stat().st_size <= 64 << 20, "Input exceeds the small-file limit: " + str(path))
    data = path.read_bytes()
    require(size is None or len(data) == size, "Pinned file size differs: " + str(path))
    require(digest is None or hashlib.sha256(data).hexdigest() == digest, "Pinned SHA-256 differs: " + str(path))
    return data


def load_helper(path):
    require(path.name == "prepare_release.py", "Use the reviewed prepare_release.py helper")
    for name, (size, digest) in HELPER_IDENTITIES.items():
        checked_bytes(path.with_name(name), size, digest)
    card_path = path.with_name("render_model_card.py").resolve()
    existing = sys.modules.get("render_model_card")
    require(existing is None or Path(existing.__file__).resolve() == card_path, "A different card helper is already imported")
    sys.path.insert(0, str(path.parent))
    card = importlib.import_module("render_model_card")
    require(Path(card.__file__).resolve() == card_path, "Imported a different card helper")
    spec = importlib.util.spec_from_file_location("spin8_notice_release_helper", path)
    require(spec is not None and spec.loader is not None, "Could not load the release helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(Path(helper.__file__).resolve() == path.resolve() and helper.card is card, "Release helper import identity differs")
    for name, (size, digest) in HELPER_IDENTITIES.items():
        checked_bytes(path.with_name(name), size, digest)
    return helper


def create_identical_or_fail(path, data):
    """Atomically create without replacing an existing directory entry."""
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(checked_bytes(path) == data, "Derived manifest already exists with different content; it was not overwritten")
    finally:
        temporary.unlink(missing_ok=True)


def derive(args):
    parent_path = args.manifest.expanduser().absolute()
    notice_path = args.notice.expanduser().absolute()
    helper_path = args.release_helper.expanduser().absolute()
    parent_bytes = checked_bytes(parent_path)
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    checked_bytes(notice_path, NOTICE_BYTES, NOTICE_SHA256)
    helper = load_helper(helper_path)
    parent = helper.check_manifest(parent_path)
    require(checked_bytes(parent_path) == parent_bytes, "Original manifest changed during validation")
    require(parent["repo_id"] == REPO_ID and parent["private"] is False, "Unexpected public destination")
    entries = {entry["path_in_repo"]: entry for entry in parent["files"]}
    require(NOTICE_NAME not in entries, "Use the original manifest without a dataset notice as input")
    required = {"benchmarks/benchmark.json", "benchmarks/benchmark.md"}
    require(required <= set(entries), "A completed benchmark JSON and Markdown must already be in the manifest")
    benchmark = helper.card.read_json(Path(entries["benchmarks/benchmark.json"]["local_path"]))
    require(benchmark.get("schema") == "spin8-gsm8k-benchmark-v1" and benchmark.get("status") == "COMPLETE",
            "Benchmark evidence is not COMPLETE")
    require(benchmark.get("dataset", {}).get("repo") == "openai/gsm8k"
            and benchmark["dataset"].get("revision") == DATASET_REVISION, "Benchmark uses a different dataset revision")
    require(benchmark.get("inputs", {}).get("candidate", {}).get("sha256") == parent["model_sha256"]
            and benchmark.get("artifacts", {}).get("benchmark_md_sha256") == entries["benchmarks/benchmark.md"]["sha256"],
            "Benchmark model or Markdown identity differs")
    samples, models = benchmark.get("protocol", {}).get("samples"), benchmark.get("models", {})
    require(type(samples) is int and samples > 0 and "candidate" in models, "Benchmark sample/model records are missing")
    for model in models.values():
        require(model.get("status") == "COMPLETE" and model.get("summary", {}).get("requested_items") == samples
                and model["summary"].get("completed_items") == samples and len(model.get("items", [])) == samples,
                "Benchmark contains incomplete model results")
    notice_entry = {"path_in_repo": NOTICE_NAME, "local_path": str(notice_path.resolve()),
                    "bytes": NOTICE_BYTES, "sha256": NOTICE_SHA256}
    derived = {**parent, "files": [*parent["files"], notice_entry],
               "total_bytes": parent["total_bytes"] + NOTICE_BYTES, "parent_manifest_sha256": parent_sha256}
    output = parent_path.with_name(OUTPUT_NAME)
    require(output != parent_path, "Derived output cannot replace the input manifest")
    data = (json.dumps(derived, indent=2, allow_nan=False) + "\n").encode()
    create_identical_or_fail(output, data)
    checked = helper.check_manifest(output)
    require(checked == derived and checked["model_sha256"] == parent["model_sha256"], "Derived manifest identity differs")
    require(checked_bytes(parent_path) == parent_bytes, "Original manifest changed; no original file was rewritten by this utility")
    checked_bytes(notice_path, NOTICE_BYTES, NOTICE_SHA256)
    return {"status": "READY", "manifest": str(output.resolve()), "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "parent_manifest_sha256": parent_sha256, "repo_id": REPO_ID, "private": False,
            "model_sha256": parent["model_sha256"], "added_path": NOTICE_NAME,
            "files": len(derived["files"]), "total_bytes": derived["total_bytes"], "publication_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("manifest", "notice", "release-helper"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(derive(args), indent=2))
        return 0
    except Exception as error:
        parser.exit(1, "Dataset notice preparation failed: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
