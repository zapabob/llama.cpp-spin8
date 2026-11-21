#!/usr/bin/env python3
"""Pinned, verified HF source download for the Spin8 Colab pipeline.

The returned directory is an HF checkpoint directory, never a GGUF filename.
Only a COMPLETE source.json may be used for conversion. Re-running a source
directory reuses its first resolved revision, including when the input was main.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tempfile
from typing import Any, Callable

QWEN_REPO = "Qwen/Qwen3.8-27B"
MATH_BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MATH_DONOR_REPO = "baseten/math-sgd-qwen38_27b-expert"
MATH_DONOR_REVISION = "5d653bbde2c8aac822f430c639b37468fe438208"
SCHEMA = "spin8-hf-source-v2"
PLAN_NAME = ".spin8-source-plan.json"
METADATA_NAMES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "chat_template.jinja", "vocab.json", "merges.txt",
    "tokenizer.model", "spiece.model", "special_tokens_map.json",
    "added_tokens.json", "model.safetensors.index.json", "README.md",
    "LICENSE", "LICENSE.txt", "NOTICE", "preprocessor_config.json",
    "video_preprocessor_config.json",
)
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2, "U32": 4, "I32": 4,
               "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def source_lock(directory: Path):
    # A kernel lock is released after a process/VM exits; no stale PID file.
    import fcntl
    with (directory / ".spin8-source.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this HF source directory") from error
        except OSError as error:
            raise RuntimeError("The source filesystem cannot lock files; use a local /content source directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _safe_relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid source file name")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != name:
        raise ValueError("Invalid source file path: " + name)
    return name


def _file_descriptor(item: Any) -> dict:
    name = _safe_relative(_field(item, "rfilename"))
    size = _field(item, "size")
    lfs = _field(item, "lfs")
    remote_sha = _field(lfs, "sha256") if lfs else None
    blob_id = _field(item, "blob_id", _field(item, "blobId"))
    if type(size) is not int or size < 0:
        raise ValueError("HF did not supply a valid file size: " + name)
    if remote_sha is not None:
        if re.fullmatch(r"[0-9a-f]{64}", remote_sha) is None:
            raise ValueError("Invalid HF LFS SHA-256: " + name)
        kind, expected = "sha256", remote_sha
    else:
        if not isinstance(blob_id, str) or re.fullmatch(r"[0-9a-f]{40}", blob_id) is None:
            raise ValueError("HF did not supply a Git blob identity: " + name)
        kind, expected = "git_blob_sha1", blob_id
    return {"path": name, "size": size, "remote_hash_kind": kind, "remote_hash": expected}


def _digest_file(path: Path, expected: dict) -> tuple[bool, dict]:
    if not path.is_file() or path.is_symlink():
        return False, {}
    size = path.stat().st_size
    if size != expected["size"]:
        return False, {"size": size}
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
            if expected["remote_hash_kind"] == "git_blob_sha1":
                blob.update(block)
    hashes = {"sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}
    good = hashes[expected["remote_hash_kind"]] == expected["remote_hash"]
    return good, {"size": size, "sha256": hashes["sha256"],
                  "remote_hash_kind": expected["remote_hash_kind"],
                  "remote_hash": expected["remote_hash"], "verified": good}


def _default_api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _default_download(**kwargs):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(**kwargs)


def _load_or_create_plan(directory: Path, repo_id: str, revision: str, token,
                         api_factory: Callable) -> tuple[dict, bool]:
    path = directory / PLAN_NAME
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("schema") != SCHEMA or plan.get("repo") != repo_id:
            raise ValueError("This source directory belongs to another repository or schema; use a new directory")
        if revision not in (plan.get("requested_revision"), plan.get("revision")):
            raise ValueError("This source directory has a different locked revision; use a new directory")
        if re.fullmatch(r"[0-9a-f]{40}", plan.get("revision", "")) is None:
            raise ValueError("The source revision lock is malformed")
        for entry in plan["inventory"]:
            _safe_relative(entry["path"])
        return plan, True
    api = api_factory(token)
    # No model search and no list_models sorting parameters are involved.
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    resolved = _field(info, "sha")
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("HF did not return a complete immutable revision")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and resolved != revision:
        raise ValueError("HF resolved a different revision than the requested immutable commit")
    siblings = _field(info, "siblings", [])
    selected = []
    for item in siblings:
        name = _field(item, "rfilename", "")
        if name in METADATA_NAMES or name.endswith(".safetensors"):
            selected.append(_file_descriptor(item))
    if len({e["path"] for e in selected}) != len(selected):
        raise ValueError("HF inventory contains duplicate source filenames")
    names = {e["path"] for e in selected}
    if "config.json" not in names or "tokenizer_config.json" not in names:
        raise ValueError("The source is missing config.json or tokenizer_config.json")
    if not ({"tokenizer.json", "tokenizer.model", "spiece.model"} & names
            or {"vocab.json", "merges.txt"} <= names):
        raise ValueError("The source has no supported tokenizer files")
    # Do not acquire ownership of conflicting pre-existing files just because
    # an earlier attempt stopped after writing a plan.
    if (directory / "source.json").exists():
        raise FileExistsError("An unowned source.json already exists; use a new local-dir")
    for entry in selected:
        existing = directory / entry["path"]
        if existing.exists() or existing.is_symlink():
            good, _ = _digest_file(existing, entry)
            if not good:
                raise FileExistsError("An unowned source file differs from HF: " + str(existing) + "; use a new local-dir")
    plan = {"schema": SCHEMA, "repo": repo_id, "requested_revision": revision,
            "revision": resolved, "inventory": sorted(selected, key=lambda e: e["path"]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "download_policy": "hf_hub_download(local_dir=...) without a second global cache copy"}
    atomic_json(path, plan)
    return plan, False


def _get_verified(directory: Path, plan: dict, expected: dict, token,
                  managed_before: bool, downloader: Callable) -> dict:
    name = _safe_relative(expected["path"])
    target = directory / name
    # Reject symlink parents as well, so a local cache cannot redirect writes.
    for parent in (target, *target.parents):
        if parent == directory.parent:
            break
        if parent.is_symlink():
            raise ValueError("Symlinks are not allowed in the managed source: " + name)
    existed = target.exists()
    if existed:
        good, record = _digest_file(target, expected)
        if good:
            return {"path": name, **record, "transfer": "verified_existing"}
        if not managed_before:
            raise FileExistsError("An unowned source file differs from HF: " + str(target) + "; use a new local-dir")
    print("Downloading " + name, flush=True)
    for attempt in range(2):
        # These arguments are supported by both current Hub 1.x and 2.x.
        result = downloader(repo_id=plan["repo"], filename=name, revision=plan["revision"],
                            local_dir=str(directory), token=token,
                            force_download=bool(existed or attempt))
        if Path(result).resolve() != target.resolve():
            raise RuntimeError("HF returned a path outside the requested local source directory")
        good, record = _digest_file(target, expected)
        if good:
            return {"path": name, **record, "transfer": "downloaded" if attempt == 0 else "redownloaded_after_hash_mismatch"}
    raise RuntimeError("Downloaded content failed its size/hash check twice: " + name)


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in Safetensors header: " + str(key))
        result[key] = value
    return result


def inspect_safetensors(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("Truncated Safetensors prefix: " + path.name)
        header_size = struct.unpack("<Q", raw_size)[0]
        if not 2 <= header_size <= 64 << 20 or header_size + 8 > size:
            raise ValueError("Invalid Safetensors header length: " + path.name)
        header = json.loads(stream.read(header_size), object_pairs_hook=_unique_json)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header is not an object: " + path.name)
    tensors, ranges = {}, []
    payload_size = size - header_size - 8
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError("Invalid Safetensors tensor descriptor: " + name)
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(shape) > 32:
            raise ValueError("Unsupported tensor dtype or shape: " + name)
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid tensor dimensions: " + name)
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError("Invalid tensor offsets: " + name)
        start, end = offsets
        elements = math.prod(shape)
        if not 0 <= start <= end <= payload_size or end - start != elements * DTYPE_BYTES[dtype]:
            raise ValueError("Tensor byte layout does not match its shape: " + name)
        tensors[name] = {"dtype": dtype, "shape": shape, "bytes": end - start}
        ranges.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError("Safetensors payload overlaps or contains holes: " + name)
        cursor = end
    if cursor != payload_size:
        raise ValueError("Safetensors payload has unreferenced or missing bytes: " + path.name)
    return {"header_bytes": header_size + 8, "payload_bytes": payload_size, "tensors": tensors}


def _select_shards(directory: Path, inventory: dict) -> tuple[list[str], dict | None, int | None]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no weight_map")
        for key, shard in weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str):
                raise ValueError("Invalid Safetensors weight_map entry")
            _safe_relative(shard)
            if not shard.endswith(".safetensors") or shard not in inventory:
                raise ValueError("Index references an unavailable Safetensors shard: " + shard)
        total = index.get("metadata", {}).get("total_size")
        if total is not None:
            if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0 or int(total) != total:
                raise ValueError("Invalid Safetensors index total_size")
            total = int(total)
        return sorted(set(weight_map.values())), weight_map, total
    if "model.safetensors" not in inventory:
        raise ValueError("Expected model.safetensors or model.safetensors.index.json")
    return ["model.safetensors"], None, None


def fetch_source(local_dir: str | Path, repo_id: str = QWEN_REPO, revision: str = "main",
                 token=None, metadata_only: bool = False, max_workers: int = 2,
                 *, _api_factory: Callable | None = None,
                 _downloader: Callable | None = None) -> dict:
    """Download and verify a pinned checkpoint, returning source.json content.

    metadata_only resolves the full plan but deliberately does not fetch weights;
    its status is METADATA_ONLY, never COMPLETE. Public models need no token.
    A supplied direction/donor is handled by the caller, not guessed here.
    """
    if not isinstance(repo_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id) is None:
        raise ValueError("Use an exact HF repository ID, such as Qwen/Qwen3.8-27B")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A revision or branch is required")
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8")
    directory = Path(local_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    api_factory = _api_factory or _default_api
    downloader = _downloader or _default_download
    with source_lock(directory):
        plan, managed_before = _load_or_create_plan(directory, repo_id, revision, token, api_factory)
        inventory = {e["path"]: e for e in plan["inventory"]}
        records = []
        report = {"schema": SCHEMA, "status": "DOWNLOADING", "repo": plan["repo"],
                  "revision": plan["revision"], "requested_revision": plan["requested_revision"],
                  "local_dir": str(directory), "files": records,
                  "source_kind": "hf_checkpoint_directory", "is_gguf": False,
                  "metadata_only": bool(metadata_only), "token_stored": False}
        atomic_json(directory / "source.json", report)
        try:
            metadata = [inventory[name] for name in METADATA_NAMES if name in inventory]
            for entry in metadata:
                if entry["size"] > 128 << 20:
                    raise ValueError("Metadata file is unexpectedly large: " + entry["path"])
                records.append(_get_verified(directory, plan, entry, token, managed_before, downloader))
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
                raise ValueError("This Spin8 workflow supports qwen3_5 source models; received " + str(config.get("model_type")))
            tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
            if "chat_template.jinja" not in inventory and not tokenizer_config.get("chat_template"):
                raise ValueError("No external or embedded chat template was downloaded")
            shards, weight_map, total_size = _select_shards(directory, inventory)
            plan["selected_shards"] = shards
            atomic_json(directory / PLAN_NAME, plan)
            report["planned_weight_files"] = [inventory[name] for name in shards]
            report["planned_weight_file_bytes"] = sum(inventory[name]["size"] for name in shards)
            report["index_tensor_count"] = len(weight_map) if weight_map is not None else None
            report["planned_weight_payload_bytes"] = total_size
            if metadata_only:
                report["status"] = "METADATA_ONLY"
                report["weight_payload_downloaded"] = False
            else:
                missing_bytes = sum(inventory[name]["size"] for name in shards
                                    if not (directory / name).is_file())
                free = shutil.disk_usage(directory).free
                if free < missing_bytes + (1 << 30):
                    raise RuntimeError(f"Insufficient source disk space: need at least {missing_bytes / 2**30:.2f} GiB plus 1 GiB reserve, have {free / 2**30:.2f} GiB")
                def fetch_weight(name):
                    return _get_verified(directory, plan, inventory[name], token, managed_before, downloader)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for record in pool.map(fetch_weight, shards):
                        records.append(record)
                        atomic_json(directory / "source.json", report)
                found = {}
                payload_bytes = 0
                header_summary = []
                for name in shards:
                    header = inspect_safetensors(directory / name)
                    for tensor, description in header["tensors"].items():
                        if tensor in found:
                            raise ValueError("Duplicate tensor across shards: " + tensor)
                        if weight_map is not None and weight_map.get(tensor) != name:
                            raise ValueError("Safetensors tensor/index ownership mismatch: " + tensor)
                        found[tensor] = description
                    payload_bytes += header["payload_bytes"]
                    header_summary.append({"path": name, "header_bytes": header["header_bytes"],
                                           "payload_bytes": header["payload_bytes"],
                                           "tensor_count": len(header["tensors"])})
                if weight_map is not None and set(found) != set(weight_map):
                    raise ValueError("Safetensors index/header tensor coverage differs")
                if total_size is not None and total_size != payload_bytes:
                    raise ValueError("Safetensors index total_size differs from all shard payloads")
                report.update(status="COMPLETE", weight_payload_downloaded=True,
                              verified_tensor_count=len(found), verified_payload_bytes=payload_bytes,
                              safetensors_headers=header_summary)
            report["completed_utc"] = datetime.now(timezone.utc).isoformat()
            report["files"] = sorted(records, key=lambda e: e["path"])
            report["content_fingerprint"] = hashlib.sha256(json.dumps(
                {"repo": report["repo"], "revision": report["revision"],
                 "files": [{"path": e["path"], "size": e["size"], "sha256": e["sha256"]}
                           for e in report["files"]]}, sort_keys=True).encode()).hexdigest()
            atomic_json(directory / "source.json", report)
            return report
        except Exception as error:
            report["status"] = "FAILED"
            report["error_type"] = type(error).__name__
            report["files"] = sorted(records, key=lambda e: e["path"])
            atomic_json(directory / "source.json", report)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=QWEN_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    report = fetch_source(args.local_dir, args.repo_id, args.revision,
                          token=os.environ.get("HF_TOKEN"), metadata_only=args.metadata_only,
                          max_workers=args.max_workers)
    print(json.dumps({"status": report["status"], "repo": report["repo"],
                      "revision": report["revision"], "local_dir": report["local_dir"],
                      "source_manifest": str(Path(report["local_dir"]) / "source.json"),
                      "verified_files": len(report["files"]),
                      "weight_payload_downloaded": report["weight_payload_downloaded"]}, indent=2))


if __name__ == "__main__":
    main()
