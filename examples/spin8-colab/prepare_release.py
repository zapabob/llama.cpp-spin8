#!/usr/bin/env python3
"""Prepare a verified public Spin8 release, or upload its exact file manifest.

Preparation is local and needs no HF credential. Upload is a separate explicit
command and requires HF_TOKEN in the environment. Neither command changes the
payload, the pipeline configuration, the native runtime or the model weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
import render_model_card as card

REPO_ID = "zapabobouj/Qwen3.8-27B-Spin8-GGUF"
SCHEMA = "spin8-release-upload-manifest-v1"


def atomic_json(path: Path, value: dict):
    card.require(not path.is_symlink(), "Refusing a symlink output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    card.require(name and not path.is_absolute() and str(path) == name
                 and all(part not in (".", "..") for part in path.parts) and "\\" not in name,
                 "Unsafe repository path: " + str(name))
    return name


def file_entry(name: str, path: Path) -> dict:
    card.require(path.is_file() and not path.is_symlink(), "Missing regular upload file: " + str(path))
    return {"path_in_repo": safe_name(name), "local_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": card.sha256(path)}


def prepare(args) -> dict:
    raw = card.read_json(args.config)
    work = Path(raw.get("work_dir", "/content/spin8-work")).expanduser().resolve()
    payload = Path(raw.get("payload_dir") or work / "payload").expanduser().resolve()
    sys.path.insert(0, str(payload))
    pipeline = importlib.import_module("colab_app.pipeline")
    card.require(Path(pipeline.__file__).resolve().parent == payload / "colab_app", "Imported a different payload")
    config = pipeline.load_config(args.config)
    card.require(config["hf_repo"] in ("", REPO_ID), "Configured HF repository differs from the requested public release")
    # Publication visibility does not enter the conversion identity. Preserve
    # the config file and use the explicitly requested public target in memory.
    config = {**config, "hf_repo": REPO_ID, "hf_private": False}
    with pipeline.pipeline_lock(config):
        build, conversion, verification = pipeline.publication_gate(config)
        file_list = pipeline.release_files(config, build, conversion, verification)
        files = dict(file_list)
        card.require(len(files) == len(file_list), "Pipeline release list contains duplicate destinations")
        release = work / "release" / conversion["config_fingerprint"]
        sources = pipeline.source_specs(config)
        card.require(len(sources) == 1 and sources[0]["role"] == "source", "This release requires the ordinary baseline")
        rendered = card.render(SimpleNamespace(
            work_dir=work, gguf=pipeline.paths(config)["output"],
            source_manifest=sources[0]["directory"] / "source.json", release_dir=release,
            benchmark=None, template=args.template))
        for entry in rendered["additional_upload_files"]:
            name, path = entry["path_in_repo"], Path(entry["local_path"])
            if name in files:
                card.require(card.sha256(files[name]) == card.sha256(path), "Conflicting release files: " + name)
            files[name] = path
        links = re.findall(r"\]\(([^)]+)\)", (release / "README.md").read_text())
        card.require(all(link.startswith(("https://", "http://", "#")) or link in files for link in links),
                     "A model-card file link is absent from the upload list")
        entries = [file_entry(name, path) for name, path in sorted(files.items())]
        model_entry = next(entry for entry in entries if entry["path_in_repo"] == config["output_name"])
        card.require(model_entry["sha256"] == conversion["output"]["sha256"], "Model changed during release preparation")
        manifest = {"schema": SCHEMA, "status": "READY", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "repo_id": REPO_ID, "repo_type": "model", "private": False,
                    "config_fingerprint": conversion["config_fingerprint"],
                    "model_file": config["output_name"], "model_sha256": model_entry["sha256"],
                    "files": entries, "total_bytes": sum(entry["bytes"] for entry in entries),
                    "publication_performed": False, "token_stored": False}
        output = release / "upload-manifest.json"
        atomic_json(output, manifest)
    return {"status": "READY", "repo_id": REPO_ID, "private": False, "manifest": str(output),
            "readme": str(release / "README.md"), "files": len(entries), "total_bytes": manifest["total_bytes"],
            "publication_performed": False}


def check_manifest(path: Path) -> dict:
    manifest = card.read_json(path)
    card.require(manifest.get("schema") == SCHEMA and manifest.get("status") == "READY"
                 and manifest.get("repo_id") == REPO_ID and manifest.get("repo_type") == "model"
                 and manifest.get("private") is False, "Manifest is not a prepared public Spin8 release")
    entries = manifest["files"]
    card.require(entries and len({entry["path_in_repo"] for entry in entries}) == len(entries), "Duplicate/empty upload list")
    for entry in entries:
        safe_name(entry["path_in_repo"])
        card.check_artifact(Path(entry["local_path"]), entry)
    names = {entry["path_in_repo"] for entry in entries}
    required = {manifest["model_file"], "README.md", "provenance.json", "conversion.json", "verification.json",
                "runtime/MANIFEST.json", "runtime/spin8-llama.patch", "runtime/colab_app/bootstrap.py",
                "runtime/licenses/LICENSE-llama.cpp", "upstream/source/README.md"}
    card.require(required <= names and any(name in names for name in ("upstream/source/LICENSE", "upstream/source/LICENSE.txt")),
                 "Release runtime/evidence/licences are incomplete")
    model = next(entry for entry in entries if entry["path_in_repo"] == manifest["model_file"])
    card.require(model["sha256"] == manifest["model_sha256"]
                 and sum(entry["bytes"] for entry in entries) == manifest["total_bytes"], "Manifest totals/model identity differ")
    return manifest


def upload(path: Path) -> dict:
    token = os.environ.get("HF_TOKEN")
    card.require(bool(token), "Upload needs HF_TOKEN with write access in this process environment")
    manifest = check_manifest(path)
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi(token=token)
    api.whoami()
    try:
        info = api.repo_info(REPO_ID, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=False)
        info = api.repo_info(REPO_ID, repo_type="model")
    card.require(not info.private, "The existing HF repository is private; no visibility change was made")
    commit = api.create_commit(repo_id=REPO_ID, repo_type="model", parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=entry["path_in_repo"], path_or_fileobj=entry["local_path"])
                    for entry in manifest["files"]], num_threads=2,
        commit_message="Publish verified Spin8 GGUF, runtime and provenance")
    report = {"schema": "spin8-manifest-publication-v1", "status": "COMMITTED", "repo": REPO_ID,
              "private": False, "commit": commit.oid, "url": str(commit.commit_url),
              "output_sha256": manifest["model_sha256"], "manifest_sha256": card.sha256(path),
              "files": [entry["path_in_repo"] for entry in manifest["files"]], "token_stored": False}
    output = path.with_name("publication.json")
    atomic_json(output, report)
    remote = api.model_info(REPO_ID, revision=commit.oid, files_metadata=True)
    card.require(not remote.private, "Committed repository is not public")
    siblings = {item.rfilename: item for item in remote.siblings}
    for entry in manifest["files"]:
        item = siblings.get(entry["path_in_repo"])
        card.require(item is not None and item.size == entry["bytes"], "Remote file size differs: " + entry["path_in_repo"])
        if item.lfs:
            digest = item.lfs["sha256"] if isinstance(item.lfs, dict) else item.lfs.sha256
            card.require(digest == entry["sha256"], "Remote LFS hash differs: " + entry["path_in_repo"])
        else:
            data = card.read_bytes(Path(entry["local_path"]))
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            card.require(item.blob_id == digest, "Remote Git blob differs: " + entry["path_in_repo"])
    report.update(status="PUBLISHED", completed_utc=datetime.now(timezone.utc).isoformat(), remote_files_verified=True)
    atomic_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Apply existing local release gates, render the card and export a file manifest")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[2] / "docs/spin8/HF_MODEL_CARD_TEMPLATE.md")
    for name, description in (("check", "Recheck every manifest file without contacting HF"),
                              ("upload", "Upload the exact public manifest using HF_TOKEN and verify the remote file identities")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "check":
            manifest = check_manifest(args.manifest)
            result = {"status": "READY", "manifest": str(args.manifest.resolve()), "files": len(manifest["files"])}
        else:
            result = upload(args.manifest)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        parser.exit(1, "Release " + args.command + " failed: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
