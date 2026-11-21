#!/usr/bin/env python3
"""Small integration tests for source pinning, download repair and tensor coverage."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest

import hf_fetch


def safetensors(tensors: dict[str, tuple[list[int], bytes]]) -> bytes:
    cursor = 0
    header = {}
    chunks = []
    for name, (shape, data) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [cursor, cursor + len(data)]}
        cursor += len(data)
        chunks.append(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


class FakeHub:
    def __init__(self):
        self.sha = "1" * 40
        self.api_calls = 0
        self.downloads = []
        self.fail_once = None
        self.files = {
            "config.json": json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 4}}).encode(),
            "tokenizer_config.json": b'{"tokenizer_class":"PreTrainedTokenizerFast"}',
            "tokenizer.json": b'{"model":{"vocab":{"a":0}}}',
            "chat_template.jinja": b'{{ messages[0].content }}',
            "model-1.safetensors": safetensors({"model.language_model.embed_tokens.weight": ([2, 4], b"\0\0" * 8)}),
            "model-2.safetensors": safetensors({"model.language_model.norm.weight": ([4], b"\x80\x3f" * 4)}),
            "assets/not-downloaded.txt": b"unrelated asset",
            "obsolete.safetensors": b"this shard is not indexed and must not be downloaded",
        }
        self.files["model.safetensors.index.json"] = json.dumps({"metadata": {"total_size": 24}, "weight_map": {
            "model.language_model.embed_tokens.weight": "model-1.safetensors",
            "model.language_model.norm.weight": "model-2.safetensors"}}).encode()

    def model_info(self, repo_id, *, revision, files_metadata):
        assert repo_id == hf_fetch.QWEN_REPO and files_metadata is True
        self.api_calls += 1
        siblings = []
        for name, data in self.files.items():
            lfs = {"sha256": hashlib.sha256(data).hexdigest()} if name.endswith(".safetensors") else None
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=blob, lfs=lfs))
        return SimpleNamespace(sha=self.sha, siblings=siblings)

    def download(self, **kwargs):
        assert set(kwargs) == {"repo_id", "filename", "revision", "local_dir", "token", "force_download"}
        name = kwargs["filename"]
        self.downloads.append((name, kwargs["revision"], kwargs["force_download"]))
        path = Path(kwargs["local_dir"]) / name
        if self.fail_once == name:
            self.fail_once = None
            raise OSError("simulated transient interruption")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        return str(path)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "source"
        self.hub = FakeHub()

    def tearDown(self):
        self.temp.cleanup()

    def fetch(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return hf_fetch.fetch_source(self.directory, _api_factory=lambda _: self.hub,
                                        _downloader=self.hub.download, max_workers=1, **kwargs)

    def test_complete_source_is_not_a_gguf(self):
        result = self.fetch()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["verified_tensor_count"], 2)
        self.assertEqual(result["verified_payload_bytes"], 24)
        self.assertFalse(result["is_gguf"])
        self.assertTrue((self.directory / "chat_template.jinja").is_file())
        self.assertFalse((self.directory / "obsolete.safetensors").exists())
        self.assertFalse((self.directory / "assets/not-downloaded.txt").exists())
        self.assertTrue(all(e["verified"] for e in result["files"]))

    def test_main_is_pinned_and_offline_rerun_uses_verified_files(self):
        first = self.fetch()
        downloads = list(self.hub.downloads)
        self.hub.sha = "2" * 40
        self.hub.download = lambda **_: (_ for _ in ()).throw(AssertionError("network should not be used"))
        second = self.fetch()
        self.assertEqual(first["revision"], second["revision"])
        self.assertEqual(first["content_fingerprint"], second["content_fingerprint"])
        self.assertEqual(self.hub.api_calls, 1)
        self.assertEqual(self.hub.downloads, downloads)

    def test_changed_explicit_revision_is_rejected(self):
        self.fetch()
        with self.assertRaisesRegex(ValueError, "different locked revision"):
            self.fetch(revision="3" * 40)

    def test_interrupted_download_resumes_and_keeps_complete_shard(self):
        self.hub.fail_once = "model-2.safetensors"
        with self.assertRaisesRegex(OSError, "transient"):
            self.fetch()
        self.assertEqual(json.loads((self.directory / "source.json").read_text())["status"], "FAILED")
        self.assertTrue((self.directory / "model-1.safetensors").is_file())
        self.hub.downloads.clear()
        result = self.fetch()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual([name for name, _, _ in self.hub.downloads], ["model-2.safetensors"])

    def test_corrupt_managed_file_is_redownloaded(self):
        self.fetch()
        path = self.directory / "model-1.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[:-1] + b"\xff")
        self.hub.downloads.clear()
        result = self.fetch()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(self.hub.downloads, [("model-1.safetensors", "1" * 40, True)])

    def test_unowned_conflicting_file_is_preserved(self):
        self.directory.mkdir()
        path = self.directory / "config.json"
        path.write_bytes(b"user data")
        with self.assertRaises(FileExistsError):
            self.fetch()
        self.assertEqual(path.read_bytes(), b"user data")
        with self.assertRaises(FileExistsError):
            self.fetch()
        self.assertFalse((self.directory / hf_fetch.PLAN_NAME).exists())
        self.assertEqual(path.read_bytes(), b"user data")

    def test_header_index_mismatch_is_not_complete(self):
        self.hub.files["model-2.safetensors"] = safetensors({"wrong.tensor": ([4], b"\0\0" * 4)})
        with self.assertRaisesRegex(ValueError, "ownership mismatch"):
            self.fetch()
        self.assertEqual(json.loads((self.directory / "source.json").read_text())["status"], "FAILED")

    def test_truncated_and_overlapping_headers_fail(self):
        self.directory.mkdir()
        target = self.directory / "bad.safetensors"
        target.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            hf_fetch.inspect_safetensors(target)
        header = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                  "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]}}
        encoded = json.dumps(header).encode()
        target.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 6)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            hf_fetch.inspect_safetensors(target)

    def test_metadata_only_never_claims_full_checkpoint(self):
        result = self.fetch(metadata_only=True)
        self.assertEqual(result["status"], "METADATA_ONLY")
        self.assertFalse(result["weight_payload_downloaded"])
        self.assertEqual(result["index_tensor_count"], 2)
        self.assertFalse((self.directory / "model-1.safetensors").exists())
        result = self.fetch()
        self.assertEqual(result["status"], "COMPLETE")

    def test_directory_lock_prevents_concurrent_mutation(self):
        self.directory.mkdir()
        with hf_fetch.source_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                self.fetch()

    def test_missing_external_and_embedded_template_stops_before_weights(self):
        del self.hub.files["chat_template.jinja"]
        with self.assertRaisesRegex(ValueError, "chat template"):
            self.fetch()
        self.assertFalse((self.directory / "model-1.safetensors").exists())


def live_metadata_test(directory: Path) -> dict:
    import huggingface_hub
    report = hf_fetch.fetch_source(directory, metadata_only=True)
    assert report["status"] == "METADATA_ONLY" and report["repo"] == hf_fetch.QWEN_REPO
    assert "chat_template.jinja" in {e["path"] for e in report["files"]}
    assert len(report["planned_weight_files"]) == 18
    second = hf_fetch.fetch_source(directory, metadata_only=True)
    assert report["content_fingerprint"] == second["content_fingerprint"]
    assert all(e["transfer"] == "verified_existing" for e in second["files"])
    return {"status": "PASS", "scope": "Actual pinned HF metadata download and verified rerun; no model tensor payload downloaded",
            "huggingface_hub": huggingface_hub.__version__,
            "repo": report["repo"], "revision": report["revision"], "files": report["files"],
            "metadata_bytes": sum(e["size"] for e in report["files"]),
            "planned_weight_bytes": report["planned_weight_file_bytes"],
            "full_model_downloaded": False, "source_manifest": str(directory / "source.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "unit_tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors)}
    if result.wasSuccessful() and args.live_dir:
        report["live"] = live_metadata_test(args.live_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
