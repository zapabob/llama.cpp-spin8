#!/usr/bin/env python3
"""Run the pinned A100 Spin8 workflow independently of the notebook kernel.

Start this process with nohup or tmux. It builds, downloads, converts, verifies
and prepares a public release manifest. It never uploads to HF. A runtime/VM
reset still ends the process; notebook reconnection alone need not do so.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
HF_REPO = "zapabobouj/Qwen3.8-27B-Spin8-GGUF"
PRECISION_KEYS = ("GGML_CUDA_SPIN8_VALIDATE_F32", "NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")

GPU_PROBE = r'''
import json, sys
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise SystemExit("A100 guard: CUDA is unavailable")
device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
if "A100" not in properties.name or properties.total_memory < 35 * 2**30:
    raise SystemExit("A100 guard: select an A100 with at least 35 GiB visible device memory")
values = torch.arange(8, device=f"cuda:{device}", dtype=torch.float32)
if (values * values).sum().item() != 140.0:
    raise SystemExit("A100 guard: CUDA arithmetic failed")
torch.cuda.synchronize(device)
report = {"status": "PASS", "name": properties.name, "device": device,
          "total_memory_bytes": properties.total_memory,
          "capability": list(torch.cuda.get_device_capability(device)),
          "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
          "cuda_arithmetic_executed": True}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
'''

RELEASE_PREPARE = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
helper = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(helper.parent))
import prepare_release
if Path(prepare_release.__file__).resolve() != helper:
    raise SystemExit("A different release helper was imported")
result = prepare_release.prepare(SimpleNamespace(config=Path(sys.argv[2]), template=Path(sys.argv[3])))
print(json.dumps(result, indent=2), flush=True)
'''


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 << 20:
        raise ValueError("Missing or invalid JSON file: " + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def atomic_json(path: Path, value: dict):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def start_ticks(pid: int):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Stopped(Exception):
    def __init__(self, signum):
        super().__init__("Runner received signal " + str(signum))
        self.signum = signum


def stop(signum, frame):
    raise Stopped(signum)


def run(args) -> int:
    config_path = args.config.expanduser().resolve()
    raw = read_json(config_path)
    work = Path(raw.get("work_dir", "/content/spin8-work")).expanduser().resolve()
    payload = Path(raw.get("payload_dir") or work / "payload").expanduser().resolve()
    sys.path.insert(0, str(payload))
    pipeline = importlib.import_module("colab_app.pipeline")
    if Path(pipeline.__file__).resolve().parent != payload / "colab_app":
        raise ValueError("A different payload was imported")
    config = pipeline.load_config(config_path)
    expected = {"source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION, "device": "cuda",
                "math_delta": False, "direction_path": "", "seed": 81237, "scale_steps": 17,
                "chunk_groups": 8192, "tile_elements": 1048576}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Use the fixed source revision and ordinary A100 cg8192/s17 baseline config")
    if config["hf_repo"] not in ("", HF_REPO):
        raise ValueError("Configured HF target differs from the requested public release")
    helper, template = args.release_helper.expanduser().resolve(), args.template.expanduser().resolve()
    inputs = [config_path, Path(__file__).resolve(), payload / "MANIFEST.json", helper, template,
              helper.with_name("render_model_card.py"),
              *(payload / "colab_app" / name for name in ("bootstrap.py", "pipeline.py", "hf_fetch.py", "hf_stream.py"))]
    if any(not path.is_file() or path.is_symlink() for path in inputs):
        raise ValueError("A required payload, helper or template file is missing")
    identities = {str(path): sha256(path) for path in inputs}
    job = (args.job_dir.expanduser().resolve() if args.job_dir else
           work / "reports/jobs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"))
    if any(job.is_relative_to(path) for path in (payload, work / "sources", work / "outputs", work / "llama.cpp")):
        raise ValueError("Keep the job directory outside payload, sources, model outputs and native source")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".spin8-release-job.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another release runner owns this work directory") from error
        job.mkdir(parents=True, exist_ok=True)
        receipt = job / "status.json"
        if receipt.exists() or (job / "job.log").exists():
            raise ValueError("Job receipts already exist; choose a new --job-dir or omit it")
        record = {"schema": "spin8-run-release-job-v1", "status": "RUNNING", "started_utc": now(),
                  "runner_pid": os.getpid(), "runner_start_ticks": start_ticks(os.getpid()),
                  "hostname": socket.gethostname(), "work_dir": str(work), "config": str(config_path),
                  "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
                  "input_sha256": identities, "stages": [], "publication_performed": False,
                  "normal_precision_environment": {name: None for name in PRECISION_KEYS}}
        boot_id = Path("/proc/sys/kernel/random/boot_id")
        if boot_id.exists():
            record["boot_id"] = boot_id.read_text().strip()
        atomic_json(receipt, record)
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        for name in PRECISION_KEYS:
            environment.pop(name, None)
        secrets = [os.environ[name] for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
                   if len(os.environ.get(name, "")) >= 8]
        console_available = True
        def redact(text):
            for secret in secrets:
                text = text.replace(secret, "[REDACTED]")
            return re.sub(r"\bhf_[A-Za-z0-9]{10,}\b", "[REDACTED]", text)
        with (job / "job.log").open("x", encoding="utf-8") as log:
            def emit(text):
                nonlocal console_available
                text = redact(text)
                log.write(text)
                log.flush()
                if console_available:
                    try:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    except BrokenPipeError:
                        console_available = False
            def stage(name, command):
                for path, digest in identities.items():
                    if sha256(Path(path)) != digest:
                        raise RuntimeError("Input changed during the job: " + path)
                entry = {"name": name, "status": "RUNNING", "started_utc": now(), "command": command}
                record["stages"].append(entry)
                record["current_stage"] = name
                atomic_json(receipt, record)
                emit("\n[" + name + "] " + entry["started_utc"] + "\n")
                started = time.monotonic()
                process = None
                try:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                        env=environment, start_new_session=True)
                    entry.update(child_pid=process.pid, child_start_ticks=start_ticks(process.pid))
                    atomic_json(receipt, record)
                    last_save = time.monotonic()
                    for line in process.stdout:
                        emit(line)
                        if time.monotonic() - last_save >= 15:
                            entry["last_output_utc"] = now()
                            atomic_json(receipt, record)
                            last_save = time.monotonic()
                    entry["returncode"] = process.wait()
                    if entry["returncode"]:
                        raise RuntimeError(name + " exited " + str(entry["returncode"]))
                    entry["status"] = "PASS"
                except BaseException as error:
                    entry["status"] = "CANCELLED" if isinstance(error, (Stopped, KeyboardInterrupt)) else "FAIL"
                    if process is not None and process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=15)
                    raise
                finally:
                    if process is not None and process.stdout is not None:
                        process.stdout.close()
                    entry.update(ended_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
                    atomic_json(receipt, record)
            emit(json.dumps({"status": "RUNNING", "runner_pid": record["runner_pid"],
                             "started_utc": record["started_utc"], "receipt": str(receipt), "log": str(job / "job.log")}) + "\n")
            python = [sys.executable, "-u", "-B"]
            try:
                stage("a100-guard", [*python, "-c", GPU_PROBE, str(job / "gpu.json")])
                record["gpu"] = read_json(job / "gpu.json")
                stage("bootstrap", [*python, str(payload / "colab_app/bootstrap.py"), "prepare",
                    "--work-dir", str(work), "--payload-dir", str(payload), "--device", "cuda", "--jobs", str(args.jobs)])
                for name in ("download", "convert", "verify"):
                    stage(name, [*python, str(payload / "colab_app/pipeline.py"), "--config", str(config_path), name])
                stage("prepare-release", [*python, "-c", RELEASE_PREPARE, str(helper), str(config_path), str(template)])
                conversion = read_json(pipeline.paths(config)["conversion"])
                manifest_path = work / "release" / conversion["config_fingerprint"] / "upload-manifest.json"
                manifest = read_json(manifest_path)
                if (manifest.get("status") != "READY" or manifest.get("repo_id") != HF_REPO
                        or manifest.get("model_sha256") != conversion["output"]["sha256"]):
                    raise RuntimeError("Release preparation did not produce the matching READY manifest")
                record.update(status="READY", completed_utc=now(), current_stage="complete",
                              upload_manifest=str(manifest_path), upload_manifest_sha256=sha256(manifest_path),
                              model=conversion["output"], repo_id=HF_REPO, private=False)
                atomic_json(receipt, record)
                emit(json.dumps({"status": "READY", "receipt": str(receipt), "upload_manifest": str(manifest_path),
                                 "publication_performed": False}, indent=2) + "\n")
                return 0
            except (Exception, KeyboardInterrupt) as error:
                cancelled = isinstance(error, (Stopped, KeyboardInterrupt))
                record.update(status="CANCELLED" if cancelled else "FAIL", ended_utc=now(), error=redact(str(error)))
                atomic_json(receipt, record)
                emit(json.dumps({"status": record["status"], "error": record["error"], "receipt": str(receipt)}) + "\n")
                return 128 + error.signum if isinstance(error, Stopped) else (130 if cancelled else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-helper", type=Path, required=True, help="Path to prepare_release.py, beside render_model_card.py")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--job-dir", type=Path, help="New receipt/log directory; defaults to WORK/reports/jobs/TIMESTAMP-PID")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        return run(args)
    except Exception as error:
        parser.exit(1, "Release job could not start: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
