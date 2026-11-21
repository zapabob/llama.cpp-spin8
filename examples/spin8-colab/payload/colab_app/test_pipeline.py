"""Pipeline integration and release gates without downloading 27B weights."""
import array
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from colab_app import bootstrap, hf_stream, pipeline as p

# Reuse the HF agent's fake HTTP inventory, while exercising its real downloader,
# remote checksums, revision locks, source manifests and Safetensors inspection.
sys.path.insert(0, str(Path(__file__).parent))
from test_hf_fetch import FakeHub


def config(tmp_path, **extra):
    file = tmp_path / "config.json"
    file.write_text(json.dumps({"work_dir": str(tmp_path / "work"), "device": "cpu",
                               "disk_reserve_gib": 1, **extra}))
    return p.load_config(file)


def hub_patch(monkeypatch):
    hub = FakeHub()
    original = p.hf_fetch.fetch_source
    def fetch(*args, **kwargs):
        return original(*args, **kwargs, _api_factory=lambda _: hub, _downloader=hub.download)
    monkeypatch.setattr(p.hf_fetch, "fetch_source", fetch)
    return hub


def fake_build(cfg):
    work = Path(cfg["work_dir"])
    repo = work / "llama.cpp"
    repo.mkdir(parents=True, exist_ok=True)
    source = repo / "source-marker.cpp"
    source.write_text("// verified fixture\n")
    report = {"status": "PASS", "device_requested": cfg["device"],
              "cuda_device_execution": cfg["device"] == "cuda", "projected_native_parity": "PASS",
              "base_commit": bootstrap.BASE_COMMIT, "patch_sha256": "f" * 64, "source_dir": str(repo),
              "source": {"verified": True, "patched_files": {source.name: bootstrap.sha256(source)}}}
    for name in ("llama-server", "test-spin8-native"):
        path = repo / name
        path.write_text("synthetic executable identity only " + name)
        report[name] = bootstrap.file_info(path)
    lib = repo / "libggml-base.so.1"
    lib.write_text("synthetic library identity")
    report["libraries"] = {lib.name: bootstrap.file_info(lib)}
    bootstrap.atomic_json(work / "reports/build-report.json", report)
    return report


def ready(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    hub = hub_patch(monkeypatch)
    with redirect_stdout(io.StringIO()):
        p.download(cfg)
    return cfg, hub, fake_build(cfg)


def fake_encoder(calls):
    def encode(source, output, **kwargs):
        calls.append({"source": source, "output": output, **kwargs})
        Path(output).write_bytes(b"GGUF" + b"fixture only, not a real model" * 4)
        return {"status": "converted", "output_sha256": bootstrap.sha256(Path(output)),
                "provenance": {"quality_measured": False, "answer_data_used": False}}
    return encode


def test_default_plain_mode_has_no_donor_or_direction_and_math_has_separate_pins(tmp_path):
    plain = config(tmp_path)
    assert plain["source_repo"] == "Qwen/Qwen3.8-27B"
    assert plain["source_revision"] == "main" and plain["math_delta"] is False
    assert plain["direction_path"] == "" and plain["hf_private"] is True
    base = p.source_specs(plain)
    assert len(base) == 1 and base[0]["revision"] == "main"
    edited = {**plain, "math_delta": True}
    specs = p.source_specs(edited)
    assert specs[0]["revision"] == p.hf_fetch.MATH_BASE_REVISION
    assert specs[1]["repo"] == p.hf_fetch.MATH_DONOR_REPO
    assert specs[1]["revision"] == p.hf_fetch.MATH_DONOR_REVISION
    assert specs[0]["directory"] != base[0]["directory"]


def test_config_rejects_tokens_path_escape_and_string_booleans(tmp_path):
    for values in ({"hf_token": "must-not-store"}, {"output_name": "../escape.gguf"},
                   {"math_delta": "false"}, {"device": "auto"}):
        with pytest.raises(ValueError):
            config(tmp_path, **values)
    assert config(tmp_path, revision="pinned-branch")["source_revision"] == "pinned-branch"


def test_real_download_manifest_then_plan_keeps_complete_status(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    hub = hub_patch(monkeypatch)
    with redirect_stdout(io.StringIO()):
        first = p.plan(cfg)
    assert first["sources"][0]["revision"] == "1" * 40
    assert first["sources"][0]["status"] == "METADATA_ONLY"
    assert not any(name.endswith(".safetensors") for name, _, _ in hub.downloads)
    with redirect_stdout(io.StringIO()):
        result = p.download(cfg)
        second = p.plan(cfg)
    assert result["status"] == "COMPLETE"
    assert second["sources"][0]["status"] == "COMPLETE"
    assert p.read_source(p.source_specs(cfg)[0])["status"] == "COMPLETE"
    assert hub.api_calls == 1
    disk = second["disk"]
    assert disk["source_weight_bytes_missing"] == 0
    assert disk["packed_cache_estimate_bytes"] > 0 and disk["gguf_estimate_bytes"] > 0
    assert disk["full_float_gguf_staged"] is False


def test_disk_gate_stops_before_weight_download(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    hub = hub_patch(monkeypatch)
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: SimpleNamespace(total=1, used=1, free=0))
    with redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="Insufficient estimated disk"):
        p.download(cfg)
    assert not any(name.endswith(".safetensors") for name, _, _ in hub.downloads)


def test_build_gate_rejects_missing_cuda_execution_and_binary_change(tmp_path):
    cfg = config(tmp_path, device="cuda")
    build = fake_build(cfg)
    path = Path(cfg["work_dir"]) / "reports/build-report.json"
    build["cuda_device_execution"] = False
    bootstrap.atomic_json(path, build)
    with pytest.raises(RuntimeError, match="mandatory CUDA"):
        p.build_gate(cfg)
    build["cuda_device_execution"] = True
    bootstrap.atomic_json(path, build)
    Path(build["llama-server"]["path"]).write_text("tampered")
    with pytest.raises(RuntimeError, match="executable changed"):
        p.build_gate(cfg)


def test_convert_uses_revision_pin_and_reuses_only_matching_output(tmp_path, monkeypatch):
    cfg, _, _ = ready(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder(calls))
    first = p.convert(cfg)
    assert first["status"] == "COMPLETE" and len(calls) == 1
    assert calls[0]["source_revision"] == "1" * 40
    assert calls[0]["donor_dir"] is None and calls[0]["direction_path"] is None
    assert calls[0]["alpha"] == 0.0 and calls[0]["device"] == "cpu"
    second = p.convert(cfg)
    assert len(calls) == 1 and second["output"]["sha256"] == first["output"]["sha256"]
    with pytest.raises(RuntimeError, match="another configuration"):
        p.convert({**cfg, "seed": cfg["seed"] + 1})
    output = p.paths(cfg)["output"]
    original = output.read_bytes()
    output.write_bytes(original[:-1] + b"x")
    with pytest.raises(RuntimeError, match="GGUF changed"):
        p.convert(cfg)
    assert output.read_bytes() == original[:-1] + b"x"


def test_unowned_gguf_is_never_overwritten(tmp_path, monkeypatch):
    cfg, _, _ = ready(tmp_path, monkeypatch)
    output = p.paths(cfg)["output"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"user existing file")
    calls = []
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder(calls))
    with pytest.raises(RuntimeError, match="conversion report"):
        p.convert(cfg)
    assert not calls and output.read_bytes() == b"user existing file"


def test_failed_conversion_records_failure_and_can_resume_same_cache(tmp_path, monkeypatch):
    cfg, _, _ = ready(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted encoder")
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fail)
    with pytest.raises(RuntimeError, match="interrupted encoder"):
        p.convert(cfg)
    previous = json.loads(p.paths(cfg)["conversion"].read_text())
    assert previous["status"] == "FAILED"
    calls = []
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder(calls))
    result = p.convert(cfg)
    assert result["status"] == "COMPLETE"
    assert str(calls[0]["cache_dir"]) == previous["cache_dir"]


def test_metadata_only_source_cannot_convert(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    hub_patch(monkeypatch)
    with redirect_stdout(io.StringIO()):
        p.plan(cfg)
    fake_build(cfg)
    with pytest.raises(RuntimeError, match="not COMPLETE"):
        p.convert(cfg)


def native_metadata(tmp_path, *, device="CPU", values=(1.0, 2.0)):
    file = tmp_path / "logits.f32"
    with file.open("wb") as handle:
        array.array("f", values).tofile(handle)
    return {"status": "PASS", "finite_logits": True, "logits_shape": [1, len(values)],
            "logits_file": str(file), "logits_dtype": "<f4" if sys.byteorder == "little" else ">f4",
            "observed_output_buffer_placements": [{"op": "MUL_MAT_SPIN8", "device": device,
                                                     "completed_observations": 4}]}


def test_native_gate_checks_real_logits_and_rejects_cpu_fallback(tmp_path):
    metadata = native_metadata(tmp_path)
    assert p.validate_native_report(metadata, "cpu")["actual_device_verified"] is True
    with pytest.raises(RuntimeError, match="placement"):
        p.validate_native_report(metadata, "cuda")
    metadata = native_metadata(tmp_path, device="CUDA0", values=(1.0, float("nan")))
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        p.validate_native_report(metadata, "cuda")


def test_verify_and_publication_gate_bind_current_model_and_runtime(tmp_path, monkeypatch):
    cfg, _, build = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder([]))
    p.convert(cfg)
    def execute(command, **kwargs):
        prefix = Path(command[command.index("--output-prefix") + 1])
        metadata = native_metadata(prefix.parent)
        bootstrap.atomic_json(prefix.with_suffix(".json"), metadata)
    monkeypatch.setattr(bootstrap, "run", execute)
    monkeypatch.setattr(bootstrap, "server_smoke", lambda *args: {"status": "PASS"})
    result = p.verify(cfg)
    assert result["status"] == "PASS" and result["quality_measured"] is False
    assert p.publication_gate(cfg)[2]["status"] == "PASS"
    verification = p.paths(cfg)["verification"]
    changed = json.loads(verification.read_text())
    changed["output"]["sha256"] = "a" * 64
    bootstrap.atomic_json(verification, changed)
    with pytest.raises(RuntimeError, match="does not match"):
        p.publication_gate(cfg)


def test_publish_requires_explicit_destination_and_write_token(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="hf_repo explicitly"):
        p.publish(cfg)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        p.publish({**cfg, "hf_repo": "owner/test"})


def test_pipeline_lock_rejects_concurrent_stage(tmp_path):
    cfg = config(tmp_path)
    with p.pipeline_lock(cfg):
        with pytest.raises(RuntimeError, match="Another pipeline stage"):
            with p.pipeline_lock(cfg):
                raise AssertionError("must not enter")


def test_release_preserves_every_manifest_file_and_roundtrips(tmp_path, monkeypatch):
    cfg, _, build = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder([]))
    conversion = p.convert(cfg)
    payload = Path(cfg["payload_dir"])
    (payload / "colab_app").mkdir(parents=True)
    (payload / "fixtures").mkdir()
    (payload / "licenses").mkdir()
    for file in Path(p.__file__).parent.glob("*.py"):
        shutil.copy2(file, payload / "colab_app" / file.name)
    (payload / "licenses/LICENSE-llama.cpp").write_text("MIT fixture")
    (payload / "spin8-llama.patch").write_text("patch fixture")
    for name in bootstrap.FIXTURES:
        (payload / "fixtures" / name).write_bytes(b"tiny fixture")
    entries = [{"path": file.relative_to(payload).as_posix(), "bytes": file.stat().st_size,
                "sha256": bootstrap.sha256(file)} for file in sorted(payload.rglob("*")) if file.is_file()]
    manifest = {"base_commit": bootstrap.BASE_COMMIT, "upstream": bootstrap.SOURCE_URL,
                "patch_sha256": bootstrap.sha256(payload / "spin8-llama.patch"), "files": entries,
                "fixture_files": [item for item in entries if item["path"].startswith("fixtures/")]}
    bootstrap.atomic_json(payload / "MANIFEST.json", manifest)
    build["patch_sha256"] = manifest["patch_sha256"]
    files = p.release_files(cfg, build, conversion, {"status": "PASS"})
    mapping = dict(files)
    assert len(mapping) == len(files)
    assert "runtime/colab_app/test_pipeline.py" in mapping
    assert "runtime/licenses/LICENSE-llama.cpp" in mapping
    restored = tmp_path / "restored-runtime"
    for name, original in files:
        if name.startswith("runtime/"):
            destination = restored / name.removeprefix("runtime/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    restored_manifest = bootstrap.validate_payload(restored)
    for item in restored_manifest["files"]:
        assert bootstrap.sha256(restored / item["path"]) == item["sha256"]
    (payload / "colab_app/test_pipeline.py").write_text("changed")
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        p.release_files(cfg, build, conversion, {"status": "PASS"})


def test_only_matching_completed_benchmark_evidence_is_included(tmp_path, monkeypatch):
    cfg, _, build = ready(tmp_path, monkeypatch)
    monkeypatch.setattr(hf_stream, "convert_hf_to_spin8", fake_encoder([]))
    conversion = p.convert(cfg)
    verification = {"status": "PASS", "config_fingerprint": conversion["config_fingerprint"],
                    "output": conversion["output"], "runtime": p.runtime_identity(build)}
    assert p.matching_benchmark_files(cfg, conversion, verification, build) == []
    directory = p.paths(cfg)["work"] / "reports/benchmarks" / p.paths(cfg)["output"].stem
    directory.mkdir(parents=True)
    markdown = directory / "benchmark.md"
    markdown.write_text("Synthetic report binding test, not a measured benchmark")
    report = {"schema": "spin8-gsm8k-benchmark-v1", "status": "COMPLETE",
              "inputs": {"candidate": conversion["output"], "server": build["llama-server"]},
              "dataset": {"repo": "openai/gsm8k", "revision": "1" * 40, "file_sha256": "2" * 64},
              "models": {"candidate": {"summary": {"requested_items": 64, "completed_items": 64,
                                                    "correct": 32, "accuracy": 0.5}}},
              "paired": {"baseline_present": False, "comparison_measured": False, "improvement_measured": False},
              "artifacts": {"benchmark_md_sha256": bootstrap.sha256(markdown)}}
    bootstrap.atomic_json(directory / "benchmark.json", report)
    accepted = p.matching_benchmark_files(cfg, conversion, verification, build)
    assert [name for name, _ in accepted] == ["benchmarks/benchmark.json", "benchmarks/benchmark.md"]
    for mutation in (lambda x: x.update(status="FAILED"),
                     lambda x: x["inputs"]["candidate"].update(sha256="3" * 64),
                     lambda x: x["inputs"]["server"].update(sha256="3" * 64),
                     lambda x: x["paired"].update(comparison_measured=True),
                     lambda x: x["models"]["candidate"]["summary"].update(completed_items=63)):
        changed = json.loads(json.dumps(report))
        mutation(changed)
        bootstrap.atomic_json(directory / "benchmark.json", changed)
        assert p.matching_benchmark_files(cfg, conversion, verification, build) == []
    bootstrap.atomic_json(directory / "benchmark.json", report)
    markdown.write_text("stale Markdown from another benchmark")
    assert p.matching_benchmark_files(cfg, conversion, verification, build) == []


def test_actual_bootstrap_native_logit_fixture_when_available():
    path = Path(__file__).resolve().parents[1] / "reports/native-parity/spin8.json"
    if not path.is_file():
        pytest.skip("Full CPU bootstrap report not present")
    result = p.validate_native_report(json.loads(path.read_text()), "cpu")
    assert result["actual_device_verified"] is True and result["logit_elements"] == 2048
