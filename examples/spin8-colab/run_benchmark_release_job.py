#!/usr/bin/env python3
"""Queue a paired GSM8K benchmark after a verified Spin8 release job is READY.

Run with nohup or subprocess.Popen(..., start_new_session=True), redirecting
stdout/stderr to a file outside --job-dir/job.log. Waiting uses only files and
Linux process identities; it does not import Torch or query the GPU. A VM reset
still ends this process. The default wait limit is six hours.

Example arguments for the current Colab work directory:
  --config /content/spin8-work/config.json
  --wait-for-job /content/spin8-work/reports/release-job-80gb/status.json
  --release-helper /content/spin8-release-tools/prepare_release.py
  --template /path/to/HF_MODEL_CARD_TEMPLATE.md
  --job-dir /content/spin8-work/reports/benchmark-release-job-80gb

The existing helpers create the F16 reference, evaluate both models and refresh
the release card/manifest. This job never uploads. Its new status.json records
the refreshed manifest hash; the original release-job receipt is never edited.
Use prepare_release.py check/upload for that refreshed manifest, rather than
rewriting the original runner receipt to make an older notebook accept it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
PRECISION_KEYS = ("GGML_CUDA_SPIN8_VALIDATE_F32", "NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")
CREDENTIAL_KEY = re.compile(r"(?:^|_)(?:TOKEN|PASSWORD|SECRET|CREDENTIALS?|API_KEY|ACCESS_KEY|PRIVATE_KEY)(?:_|$)", re.I)
MANIFEST_SHA256 = "5566b8203cb11d6478fe80de710d2929d886fcb08a03f857ecac0829eb01582a"
SOURCE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
REPO_ID = "zapabobouj/Qwen3.8-27B-Spin8-GGUF"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 << 20,
            "Missing or invalid JSON: " + str(path))
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "Expected a JSON object: " + str(path))
    return value


def atomic_json(path, value):
    require(not path.is_symlink(), "Refusing a symlink receipt")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_info(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rpartition(")")[2].split()
        return {"start_ticks": int(fields[19]), "state": fields[0], "group": int(fields[2])}
    except (OSError, ValueError, TypeError, IndexError):
        return None


def process_matches(pid, ticks, recorded_boot):
    info = process_info(pid)
    return (recorded_boot == boot_id() and ticks is not None and info is not None
            and info["start_ticks"] == ticks and info["state"] not in ("Z", "X"))


def source_identity(record):
    return {key: record.get(key) for key in
            ("schema", "started_utc", "runner_pid", "runner_start_ticks", "boot_id", "work_dir", "config")}


class NoLiveProcess(RuntimeError):
    pass


def source_state(record, expected):
    require(source_identity(record) == expected, "The original release-job identity changed")
    require(record.get("boot_id") == boot_id(), "Original release job belongs to another VM boot")
    status = record.get("status")
    require(status not in ("FAIL", "CANCELLED"), "Original release job ended with " + str(status))
    if status == "READY":
        return "READY"
    require(status == "RUNNING", "Unexpected original release-job status: " + str(status))
    live = process_matches(record.get("runner_pid"), record.get("runner_start_ticks"), record.get("boot_id"))
    for stage in record.get("stages", []):
        if stage.get("status") == "RUNNING":
            live = live or process_matches(stage.get("child_pid"), stage.get("child_start_ticks"), record.get("boot_id"))
    if not live:
        raise NoLiveProcess("Original release job is not READY and has no verifiable live runner or current child")
    return "RUNNING"


def group_active(group):
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_info(path.name)
            if info and info["group"] == group and info["state"] not in ("Z", "X"):
                return True
    return False


def terminate_group(process, grace=15):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while group_active(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    if group_active(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while group_active(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    require(not group_active(process.pid), "Owned child process group did not stop after SIGKILL")
    process.wait(timeout=5)


def clean_environment():
    environment = {key: value for key, value in os.environ.items()
                   if key not in PRECISION_KEYS and not CREDENTIAL_KEY.search(key)}
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    return environment


def pin_inputs(config_path, payload, helper, template):
    manifest_path = payload / "MANIFEST.json"
    require(sha256(manifest_path) == MANIFEST_SHA256, "Payload manifest differs from the reviewed A100 snapshot")
    manifest = read_json(manifest_path)
    files = [config_path, Path(__file__).resolve(), manifest_path, helper, helper.with_name("render_model_card.py"), template]
    names = set()
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        require(name and not relative.is_absolute() and str(relative) == name
                and ".." not in relative.parts and "\\" not in name and name not in names, "Invalid payload inventory")
        names.add(name)
        path = payload / name
        require(path.resolve().is_relative_to(payload) and path.is_file() and not path.is_symlink()
                and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"], "Payload file differs: " + name)
        files.append(path)
    require({"colab_app/baseline.py", "colab_app/benchmark.py"} <= names, "Payload lacks benchmark helpers")
    require(all(path.is_file() and not path.is_symlink() for path in files), "A pinned input is missing")
    return {str(path): sha256(path) for path in files}


@contextmanager
def work_lock(path, timeout=0):
    with path.open("a+") as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, "Work lock is still held: " + str(path))
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        yield


class Stopped(Exception):
    def __init__(self, signum):
        super().__init__("Benchmark runner received signal " + str(signum))
        self.signum = signum


def stop(signum, frame):
    raise Stopped(signum)


class Job:
    def __init__(self, directory, record):
        require(not (directory / "status.json").exists() and not (directory / "job.log").exists(),
                "Job receipts already exist; choose a new --job-dir")
        directory.mkdir(parents=True, exist_ok=True)
        self.receipt, self.record = directory / "status.json", record
        self.log = (directory / "job.log").open("x", encoding="utf-8")
        self.secrets = [value for key, value in os.environ.items() if CREDENTIAL_KEY.search(key) and len(value) >= 8]
        self.console = True
        self.save()

    def save(self):
        atomic_json(self.receipt, self.record)

    def redact(self, text):
        for value in self.secrets:
            text = text.replace(value, "[REDACTED]")
        return re.sub(r"\bhf_[A-Za-z0-9]{10,}\b", "[REDACTED]", text)

    def emit(self, text):
        text = self.redact(text)
        self.log.write(text + "\n")
        self.log.flush()
        if self.console:
            try:
                print(text, flush=True)
            except BrokenPipeError:
                self.console = False

    def check_inputs(self):
        for path, digest in self.record["input_sha256"].items():
            require(sha256(path) == digest, "Pinned input changed: " + path)

    def stage(self, name, command, environment):
        self.check_inputs()
        entry = {"name": name, "status": "RUNNING", "started_utc": now(), "command": command}
        self.record["stages"].append(entry)
        self.record.update(status="RUNNING", current_stage=name)
        self.save()
        self.emit("[" + name + "] " + entry["started_utc"])
        process = None
        started = time.monotonic()
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                env=environment, start_new_session=True)
            info = process_info(process.pid)
            entry.update(child_pid=process.pid, child_start_ticks=info["start_ticks"] if info else None)
            self.save()
            last_save = time.monotonic()
            for line in process.stdout:
                self.emit(line.rstrip("\n"))
                if time.monotonic() - last_save >= 15:
                    entry["last_output_utc"] = now()
                    self.save()
                    last_save = time.monotonic()
            entry["returncode"] = process.wait()
            require(entry["returncode"] == 0, name + " exited " + str(entry["returncode"]))
            entry["status"] = "PASS"
        except BaseException as error:
            entry["status"] = "CANCELLED" if isinstance(error, (Stopped, KeyboardInterrupt)) else "FAIL"
            if process is not None:
                terminate_group(process)
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            entry.update(ended_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
            self.save()


def wait_for_release(job, path, timeout, poll):
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        require(time.monotonic() < deadline, "Timed out waiting for the original release job")
        record = read_json(path)
        try:
            state = source_state(record, job.record["original_job_identity"])
        except NoLiveProcess:
            # A final RUNNING snapshot can race the atomic READY write and exit.
            record = read_json(path)
            require(source_state(record, job.record["original_job_identity"]) == "READY",
                    "Original process exited without a matching READY receipt")
            return record
        if state == "READY":
            return record
        stage = record.get("current_stage")
        job.record.update(current_stage="waiting-for-release", last_checked_utc=now(), original_current_stage=stage)
        job.save()
        if stage != previous:
            job.emit("Waiting for original release job: " + str(stage))
            previous = stage
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))


def validate_benchmark(report, baseline, build, conversion, args, benchmark, card, directory):
    require(report.get("schema") == benchmark.SCHEMA and report.get("status") == "COMPLETE", "Benchmark is not COMPLETE")
    expected = {"candidate": conversion["output"], "server": build["llama-server"], "baseline": baseline["output"]}
    for label, artifact in expected.items():
        actual = report["inputs"].get(label, {})
        require(actual.get("sha256") == artifact["sha256"] and actual.get("bytes") == artifact["bytes"]
                and Path(actual.get("path", "")).resolve() == Path(artifact["path"]).resolve(), "Benchmark input differs: " + label)
    protocol = {"samples": args.samples, "seed": args.seed, "max_tokens": args.max_tokens, "ctx_size": args.ctx,
                "thinking": "off", "gpu_layers": "99", "baseline_gpu_layers": "99", "threads": 4,
                "load_timeout": 900, "request_timeout": 1200, "vram_interval": 0.2,
                "temperature": 0.0, "few_shot": 0, "training_performed": False, "test_answers_in_prompts": False}
    require(all(report["protocol"].get(key) == value for key, value in protocol.items()), "Benchmark protocol differs")
    require(report.get("benchmark_source_sha256") == sha256(Path(benchmark.__file__)), "Benchmark code identity differs")
    dataset = report["dataset"]
    require(dataset.get("repo") == benchmark.DATASET_REPO and dataset.get("revision") == benchmark.DATASET_REVISION
            and dataset.get("file_sha256") == benchmark.DATASET_SHA256, "Dataset identity differs")
    require(set(report["models"]) == {"baseline", "candidate"}, "Both model results are required")
    for result in report["models"].values():
        require(result.get("status") == "COMPLETE" and result["summary"]["requested_items"] == args.samples
                and result["summary"]["completed_items"] == args.samples and len(result["items"]) == args.samples,
                "A model result is incomplete")
    paired = benchmark.paired_results(report["models"], args.samples)
    require(paired == report["paired"] and paired.get("baseline_present") is True
            and paired.get("comparison_measured") is True and paired.get("paired_items") == args.samples,
            "Paired comparison is incomplete or inconsistent")
    card.optional_benchmark(directory / "benchmark.json", conversion["output"]["sha256"], build["llama-server"]["sha256"], {})


def run(args):
    config_path, original_path = args.config.expanduser().resolve(), args.wait_for_job.expanduser().resolve()
    raw = read_json(config_path)
    work = Path(raw.get("work_dir", "/content/spin8-work")).expanduser().resolve()
    payload = Path(raw.get("payload_dir") or work / "payload").expanduser().resolve()
    helper, template, directory = args.release_helper.expanduser().resolve(), args.template.expanduser().resolve(), args.job_dir.expanduser().resolve()
    require(not any(directory.is_relative_to(path) for path in
                    (payload, work / "llama.cpp", work / "sources", work / "outputs", work / "release", original_path.parent)),
            "Keep new job receipts outside source, outputs, payload and the original job directory")
    original = read_json(original_path)
    require(original.get("schema") == "spin8-run-release-job-v1" and Path(original["config"]).resolve() == config_path
            and Path(original["work_dir"]).resolve() == work, "Original job does not match this configuration")
    identities = pin_inputs(config_path, payload, helper, template)
    require(original.get("input_sha256", {}).get(str(config_path)) == identities[str(config_path)]
            and original.get("input_sha256", {}).get(str(payload / "MANIFEST.json")) == MANIFEST_SHA256,
            "Original job config/payload identity differs")
    info = process_info(os.getpid())
    job = Job(directory, {"schema": "spin8-benchmark-release-job-v1", "status": "WAITING", "started_utc": now(),
        "runner_pid": os.getpid(), "runner_start_ticks": info["start_ticks"], "boot_id": boot_id(), "hostname": socket.gethostname(),
        "work_dir": str(work), "config": str(config_path), "wait_for_job": str(original_path),
        "original_job_identity": source_identity(original), "input_sha256": identities, "stages": [],
        "samples": args.samples, "seed": args.seed, "max_tokens": args.max_tokens, "ctx": args.ctx,
        "publication_performed": False, "normal_precision_environment": {name: None for name in PRECISION_KEYS}})
    environment = clean_environment()
    python = [sys.executable, "-u", "-B"]
    try:
        wait_for_release(job, original_path, args.wait_timeout, args.poll_seconds)
        with work_lock(work / ".spin8-benchmark-release-job.lock"), work_lock(work / ".spin8-release-job.lock", timeout=60):
            original = read_json(original_path)
            require(source_state(original, job.record["original_job_identity"]) == "READY", "Original job is no longer READY")
            job.check_inputs()
            job.record["original_ready_receipt_sha256"] = sha256(original_path)
            old_manifest = Path(original["upload_manifest"]).resolve()
            require(old_manifest.is_relative_to(work / "release") and sha256(old_manifest) == original["upload_manifest_sha256"],
                    "Original READY manifest changed before the benchmark job")
            job.record["original_upload_manifest_sha256"] = original["upload_manifest_sha256"]
            sys.path.insert(0, str(helper.parent))
            release = importlib.import_module(helper.stem)
            require(Path(release.__file__).resolve() == helper, "Imported a different release helper")
            card = release.card
            require(Path(card.__file__).resolve() == helper.with_name("render_model_card.py"), "Imported a different card helper")
            sys.path.insert(0, str(payload))
            pipeline = importlib.import_module("colab_app.pipeline")
            benchmark = importlib.import_module("colab_app.benchmark")
            require(Path(pipeline.__file__).resolve().parent == payload / "colab_app"
                    and Path(benchmark.__file__).resolve() == payload / "colab_app/benchmark.py", "Imported a different payload")
            config = pipeline.load_config(config_path)
            require(config["source_repo"] == "Qwen/Qwen3.8-27B" and config["source_revision"] == SOURCE_REVISION
                    and config["device"] == "cuda" and config["math_delta"] is False and not config["direction_path"],
                    "This queued release requires the ordinary pinned CUDA baseline")
            with pipeline.pipeline_lock(config):
                build, conversion, verification = pipeline.publication_gate(config)
            locations = pipeline.paths(config)
            require(conversion["output"] == original["model"], "Current candidate differs from the original READY job")
            baseline_path = work / "reports/baseline-f16.json"
            job.stage("baseline-f16", [*python, str(payload / "colab_app/baseline.py"), "--config", str(config_path),
                                      "--report", str(baseline_path)], {**environment, "CUDA_VISIBLE_DEVICES": ""})
            baseline = read_json(baseline_path)
            require(baseline.get("schema") == "spin8-original-f16-baseline-v1" and baseline.get("status") == "COMPLETE",
                    "F16 baseline did not complete; a candidate-only fallback is not allowed")
            require(baseline.get("candidate_sha256") == conversion["output"]["sha256"],
                    "Baseline receipt was prepared for another candidate")
            source = pipeline.read_source(pipeline.source_specs(config)[0])
            expected = {"source_repo": source["repo"], "source_revision": source["revision"],
                        "source_content_fingerprint": source["content_fingerprint"], "outtype": "f16",
                        "no_mtp": True, "unmodified_source": True}
            require(all(baseline["identity"].get(key) == value for key, value in expected.items())
                    and baseline["fingerprint"] == pipeline.digest_json(baseline["identity"]), "Baseline source identity differs")
            baseline_gguf = Path(baseline["output"]["path"]).resolve()
            require(baseline_gguf.is_relative_to(work / "baselines"), "Baseline output is outside the work directory")
            card.check_artifact(baseline_gguf, baseline["output"])
            output_dir = work / "reports/benchmarks" / locations["output"].stem
            command = [*python, str(payload / "colab_app/benchmark.py"), "--server-bin", build["llama-server"]["path"],
                "--candidate-gguf", str(locations["output"]), "--baseline-gguf", str(baseline_gguf), "--output-dir", str(output_dir),
                "--dataset-dir", str(work / "datasets/gsm8k"),
                "--samples", str(args.samples), "--seed", str(args.seed), "--max-tokens", str(args.max_tokens),
                "--ctx-size", str(args.ctx), "--thinking", "off", "--gpu-layers", "99", "--baseline-gpu-layers", "99",
                "--threads", "4", "--load-timeout", "900", "--request-timeout", "1200", "--vram-interval", "0.2", "--install-missing"]
            with pipeline.pipeline_lock(config):
                job.stage("paired-gsm8k", command, environment)
                report_path = output_dir / "benchmark.json"
                report = read_json(report_path)
                validate_benchmark(report, baseline, build, conversion, args, benchmark, card, output_dir)
                required = {"benchmarks/benchmark.json", "benchmarks/benchmark.md"}
                require({name for name, _ in pipeline.matching_benchmark_files(config, conversion, verification, build)} == required,
                        "Existing release gates did not accept both benchmark evidence files")
            report_sha = sha256(report_path)
            job.record["benchmark"] = {"path": str(report_path), "sha256": report_sha,
                "baseline_correct": report["models"]["baseline"]["summary"]["correct"],
                "candidate_correct": report["models"]["candidate"]["summary"]["correct"], "samples": args.samples}
            job.stage("prepare-benchmarked-release", [*python, str(helper), "prepare", "--config", str(config_path),
                                                      "--template", str(template)], environment)
            manifest_path = work / "release" / conversion["config_fingerprint"] / "upload-manifest.json"
            manifest = release.check_manifest(manifest_path)
            entries = {entry["path_in_repo"]: entry for entry in manifest["files"]}
            require(manifest["repo_id"] == REPO_ID and manifest["model_sha256"] == conversion["output"]["sha256"]
                    and required <= set(entries) and entries["benchmarks/benchmark.json"]["sha256"] == report_sha
                    and entries["benchmarks/benchmark.md"]["sha256"] == report["artifacts"]["benchmark_md_sha256"],
                    "Refreshed release manifest omitted or changed benchmark evidence")
            job.check_inputs()
            require(sha256(original_path) == job.record["original_ready_receipt_sha256"], "Original runner receipt changed")
            job.record.update(status="READY", current_stage="complete", completed_utc=now(),
                upload_manifest=str(manifest_path), upload_manifest_sha256=sha256(manifest_path),
                model=conversion["output"], repo_id=REPO_ID, private=False)
            job.save()
            job.emit(json.dumps({"status": "READY", "receipt": str(job.receipt), "benchmark": job.record["benchmark"],
                                 "upload_manifest": str(manifest_path), "publication_performed": False}))
            return 0
    except (Exception, KeyboardInterrupt) as error:
        cancelled = isinstance(error, (Stopped, KeyboardInterrupt))
        job.record.update(status="CANCELLED" if cancelled else "FAIL", ended_utc=now(), error=job.redact(str(error)))
        job.save()
        job.emit(json.dumps({"status": job.record["status"], "error": job.record["error"], "receipt": str(job.receipt)}))
        return 128 + error.signum if isinstance(error, Stopped) else (130 if cancelled else 1)
    finally:
        job.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("config", "wait-for-job", "release-helper", "template", "job-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--ctx", "--ctx-size", dest="ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--wait-timeout", type=float, default=21600, help="Seconds to wait for READY; at most 86400")
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if (not 1 <= args.samples <= 1319 or args.max_tokens < 1 or args.ctx <= args.max_tokens or args.seed < 0
            or not math.isfinite(args.wait_timeout) or not 0 < args.wait_timeout <= 86400
            or not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60):
        parser.error("Invalid sample count, context/token budget, seed or bounded waiting interval")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        return run(args)
    except Exception as error:
        parser.exit(1, "Benchmark job could not start: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
