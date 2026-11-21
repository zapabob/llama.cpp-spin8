"""Regression checks for the previous Colab install/build failure modes."""
from argparse import Namespace
import json
import shutil
import subprocess
import sys

import pytest

from colab_app import bootstrap as b


def clean_probe():
    return {"modules": {name: {"status": "ok"} for name in ["torch", *b.REQUIRED]},
            "distributions": {"torch": "2.14.1+cu130", "numpy": "2.4.6"}}


def test_only_missing_auxiliary_is_installed_and_torch_never_replaced():
    probe = clean_probe()
    assert b.dependency_plan(probe) == []
    probe["modules"]["sentencepiece"] = {"status": "missing"}
    assert b.dependency_plan(probe) == [b.REQUIRED["sentencepiece"]]
    probe["modules"]["torch"] = {"status": "missing"}
    with pytest.raises(RuntimeError, match="never replaces PyTorch"):
        b.dependency_plan(probe)


def test_installed_broken_import_does_not_trigger_dependency_overwrite():
    probe = clean_probe()
    probe["modules"]["scipy.linalg"] = {"status": "broken", "error": "numpy ABI mismatch"}
    with pytest.raises(RuntimeError, match="numpy ABI mismatch"):
        b.dependency_plan(probe)


@pytest.mark.parametrize(("capability", "architecture"), [([7, 5], "75"), ([8, 0], "80"),
                                                          ([9, 0], "90"), ([10, 0], "100")])
def test_architecture_follows_actual_device(capability, architecture):
    assert b.cuda_architecture({"gpu": {"capability": capability,
                                        "torch_device_executed": True}}) == architecture


def test_cuda_availability_without_execution_is_not_sufficient(tmp_path):
    with pytest.raises(RuntimeError, match="actual CUDA tensor operation"):
        b.cuda_architecture({"gpu": {"capability": [8, 0], "torch_device_executed": False}})
    path = tmp_path / "cuda.json"
    path.write_text(json.dumps({"cuda_executed": False, "status": "skipped_no_cuda_device"}))
    with pytest.raises(RuntimeError, match="skipped or failed"):
        b.require_cuda_execution(path)


def test_cpu_fallback_is_not_a_cuda_parity_pass(tmp_path):
    path = tmp_path / "parity.json"
    path.write_text(json.dumps({"status": "PASS", "comparison": {
        "all_observed_required_op_output_devices_match": False}}))
    with pytest.raises(RuntimeError, match="actual output device"):
        b.require_parity(path)


def test_payload_path_cannot_escape_its_root(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "link").symlink_to(outside)
    for name in ("../outside", str(outside), "link", "dir/../../outside"):
        with pytest.raises(ValueError):
            b.safe_payload_file(payload, name)


def test_failed_stage_report_is_concrete_and_not_pass(tmp_path):
    work = tmp_path / "work"
    payload = tmp_path / "payload"
    payload.mkdir()
    args = Namespace(work_dir=work, payload_dir=payload, device="cpu", jobs=1,
                     with_ui=False, ui_version="latest")
    with pytest.raises(ValueError, match="MANIFEST.json"):
        b.prepare(args)
    report = json.loads((work / "reports/build-report.json").read_text())
    assert report["status"] == "FAIL"
    assert report["current_stage"] == "validate payload"
    assert report["cuda_device_execution"] is False
    assert report["stages"][0]["status"] == "FAIL"
    assert not (work / "llama.cpp").exists()


def test_compiler_output_is_live_and_failed_command_keeps_log(tmp_path, capsys):
    log = tmp_path / "command.log"
    command = [sys.executable, "-u", "-c",
               "import time; print('compile-start', flush=True); time.sleep(.15); print('compile-error'); raise SystemExit(7)"]
    with pytest.raises(RuntimeError, match="exited 7"):
        b.run(command, log=log, heartbeat=0.04)
    output = capsys.readouterr().out
    assert "compile-start" in output and "Still running" in output and "compile-error" in output
    assert "compile-start" in log.read_text() and "compile-error" in log.read_text()


def test_complete_source_verification_rejects_unrelated_edit(tmp_path, monkeypatch):
    git = shutil.which("git")
    if not git:
        pytest.skip("git is unavailable")
    repo = tmp_path / "repo"
    payload = tmp_path / "payload"
    reports = tmp_path / "reports"
    for path in (repo, payload, reports):
        path.mkdir()
    subprocess.run([git, "init", str(repo)], check=True, capture_output=True)
    subprocess.run([git, "remote", "add", "origin", b.SOURCE_URL], cwd=repo, check=True)
    (repo / "a.txt").write_text("before\nunchanged\n")
    subprocess.run([git, "add", "a.txt"], cwd=repo, check=True)
    subprocess.run([git, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"],
                   cwd=repo, check=True, capture_output=True)
    base = b.capture([git, "rev-parse", "HEAD"], cwd=repo)
    monkeypatch.setattr(b, "BASE_COMMIT", base)
    (repo / "a.txt").write_text("after\nunchanged\n")
    (payload / "spin8-llama.patch").write_text(b.capture([git, "diff", "--binary", "HEAD"], cwd=repo) + "\n")
    assert b.prepare_source(repo, payload, reports, git)["verified"] is True
    (repo / "unrelated.txt").write_text("preserve this user file")
    with pytest.raises(RuntimeError, match="additional source changes"):
        b.prepare_source(repo, payload, reports, git)
    assert (repo / "unrelated.txt").read_text() == "preserve this user file"
    # Even a change within a legitimate patched file must fail exact verification.
    (repo / "unrelated.txt").unlink()
    (repo / "a.txt").write_text("after\nunchanged\nextra user content\n")
    with pytest.raises(RuntimeError, match="contents differ"):
        b.prepare_source(repo, payload, reports, git)
    assert "extra user content" in (repo / "a.txt").read_text()
