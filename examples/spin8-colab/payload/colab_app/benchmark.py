#!/usr/bin/env python3
"""Reproducible GSM8K screening for two sequential native llama-server runs."""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

SCHEMA = "spin8-gsm8k-benchmark-v1"
DATASET_REPO = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_FILE = "main/test-00000-of-00001.parquet"
DATASET_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
DATASET_BYTES = 419088
DATASET_ROWS = 1319
PYARROW_VERSION = "26.0.1"
SEED = 20261121
INSTRUCTION = (
    "Solve this math word problem. Explain your calculation, then end your response "
    "with a separate line in exactly this format: #### <number>. "
    "The final line must contain only the number after ####, without units.\n\n"
)
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
SOURCES = [
    "https://huggingface.co/datasets/openai/gsm8k/tree/" + DATASET_REVISION,
    "https://github.com/openai/grade-school-math",
    "https://github.com/ggml-org/llama.cpp/blob/7fce5439fe451cd48340b112ec37d2befe02c776/tools/server/README.md",
    "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextlib.contextmanager
def run_lock(directory: Path):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".benchmark.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark owns this output directory") from error
        yield


def numeric_value(value: str) -> str | None:
    value = value.strip()
    if len(value) > 128 or not re.fullmatch(NUMBER, value):
        return None
    try:
        number = Decimal(value.replace(",", ""))
        if not number.is_finite() or abs(number.adjusted()) > 1000:
            return None
        return "0" if number == 0 else str(number.normalize())
    except InvalidOperation:
        return None


def final_answer(text: str, *, starts_in_thinking: bool = False) -> dict:
    """Score only the final answer, never numbers inside an unfinished thought."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif starts_in_thinking:
        text = ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    strict = re.fullmatch(r"####\s*(" + NUMBER + r")", last)
    if strict:
        value = numeric_value(strict.group(1))
        return {"value": value, "format_ok": value is not None, "method": "hash_final_line", "final_text": text}
    boxed = re.search(r"\\boxed\{\s*(" + NUMBER + r")\s*\}\s*[.$]?\s*$", last)
    if boxed:
        return {"value": numeric_value(boxed.group(1)), "format_ok": False, "method": "boxed_final", "final_text": text}
    plain = re.fullmatch(r"(?:(?:final\s+answer|answer|the\s+answer\s+is)\s*[:=]?\s*)?\$?(" + NUMBER + r")\.?",
                         last, flags=re.IGNORECASE)
    value = numeric_value(plain.group(1)) if plain else None
    return {"value": value, "format_ok": False,
            "method": "plain_final_line" if value is not None else "unextractable", "final_text": text}


def fetch_dataset(directory: Path, *, install_missing: bool = False) -> tuple[list[dict], dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        if not install_missing:
            raise RuntimeError("GSM8K needs pyarrow. Rerun with --install-missing; this installs only missing pyarrow, not Torch.") from error
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-deps", "pyarrow==" + PYARROW_VERSION], check=True)
        import pyarrow.parquet as parquet
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_FILE
    if not path.is_file() or path.stat().st_size != DATASET_BYTES or sha256(path) != DATASET_SHA256:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
                                   filename=DATASET_FILE, local_dir=directory, force_download=path.exists()))
    if path.stat().st_size != DATASET_BYTES or sha256(path) != DATASET_SHA256:
        raise ValueError("The pinned GSM8K test file failed its published SHA256/size check")
    table = parquet.read_table(path, columns=["question", "answer"])
    if table.num_rows != DATASET_ROWS:
        raise ValueError("Unexpected pinned GSM8K test row count")
    items = []
    for index, row in enumerate(table.to_pylist()):
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip():
            raise ValueError("Invalid GSM8K question/answer row")
        gold = final_answer(answer)
        if not gold["format_ok"]:
            raise ValueError("Invalid gold final answer at GSM8K row " + str(index))
        items.append({"item_id": index, "question": question, "gold": gold["value"]})
    manifest = {"repo": DATASET_REPO, "revision": DATASET_REVISION, "subset": "main", "split": "test",
                "file": DATASET_FILE, "file_sha256": DATASET_SHA256, "file_bytes": DATASET_BYTES,
                "total_items": DATASET_ROWS, "local_file": str(path.resolve()),
                "pyarrow_version": importlib.metadata.version("pyarrow"), "training_split_used": False}
    atomic_json(directory / "dataset-lock.json", manifest)
    return items, manifest


def select_items(items: list[dict], count: int, seed: int) -> list[dict]:
    if not 1 <= count <= len(items):
        raise ValueError("Sample count must be between 1 and the test set size")
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item['item_id']}".encode()).digest())[:count]


def wilson_interval(correct: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    p, z2 = correct / count, z * z
    centre = (p + z2 / (2 * count)) / (1 + z2 / count)
    half = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / (1 + z2 / count)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(items: list[dict], requested: int) -> dict:
    count = len(items)
    correct = sum(item["correct"] for item in items)
    strict = sum(item["correct"] and item["format_ok"] for item in items)
    formats = sum(not item["format_ok"] for item in items)
    extraction = sum(item["prediction"] is None for item in items)
    result = {"requested_items": requested, "completed_items": count, "correct": correct,
              "accuracy": correct / count if count else None, "accuracy_wilson95": wilson_interval(correct, count),
              "strict_format_correct": strict, "strict_format_accuracy": strict / count if count else None,
              "format_failures": formats, "format_failure_rate": formats / count if count else None,
              "extraction_failures": extraction, "extraction_failure_rate": extraction / count if count else None,
              "token_limit_count": sum(item["stop_type"] == "limit" for item in items),
              "context_truncation_count": sum(item["context_truncated"] for item in items),
              "request_wall_seconds": sum(item["request_seconds"] for item in items)}
    for name, prefix in (("prompt", "prompt"), ("decode", "predicted")):
        usable = [item["timings"] for item in items
                  if isinstance(item["timings"].get(prefix + "_n"), (int, float))
                  and isinstance(item["timings"].get(prefix + "_ms"), (int, float))
                  and math.isfinite(item["timings"][prefix + "_ms"])
                  and item["timings"][prefix + "_ms"] > 0
                  and math.isfinite(item["timings"][prefix + "_n"])
                  and item["timings"][prefix + "_n"] >= 0
                  and isinstance(item["timings"].get(prefix + "_per_second"), (int, float))
                  and math.isfinite(item["timings"][prefix + "_per_second"])
                  and item["timings"][prefix + "_per_second"] >= 0]
        tokens = sum(value[prefix + "_n"] for value in usable)
        seconds = sum(value[prefix + "_ms"] for value in usable) / 1000
        # The server's decode clock excludes the first token; predicted_n includes it.
        timed_tokens = sum(value[prefix + "_per_second"] * value[prefix + "_ms"] / 1000 for value in usable)
        result[name + "_tokens"] = tokens
        result[name + "_timed_tokens"] = timed_tokens
        result[name + "_seconds"] = seconds
        result[name + "_timing_items"] = len(usable)
        result[name + "_tokens_per_second"] = timed_tokens / seconds if seconds else None
    result["cached_prompt_tokens"] = sum(item["timings"].get("cache_n", 0) for item in items)
    return result


def paired_results(models: dict, requested: int) -> dict:
    base = models.get("baseline", {}).get("items", [])
    candidate = models.get("candidate", {}).get("items", [])
    base_map, candidate_map = ({item["item_id"]: item for item in values} for values in (base, candidate))
    shared = sorted(base_map.keys() & candidate_map.keys())
    pairs = []
    for index in shared:
        left, right = base_map[index], candidate_map[index]
        if left["prompt_sha256"] != right["prompt_sha256"] or left["prompt_tokens_sha256"] != right["prompt_tokens_sha256"]:
            raise ValueError("Paired prompt or tokenizer mismatch at item " + str(index))
        pairs.append({"item_id": index, "baseline_correct": left["correct"], "candidate_correct": right["correct"],
                      "baseline_prediction": left["prediction"], "candidate_prediction": right["prediction"],
                      "gold": left["gold"]})
    wins = sum(not item["baseline_correct"] and item["candidate_correct"] for item in pairs)
    losses = sum(item["baseline_correct"] and not item["candidate_correct"] for item in pairs)
    complete = len(pairs) == requested and all(value.get("status") == "COMPLETE" for value in models.values())
    placements = [models.get(label, {}).get("device_placement", {}) for label in ("baseline", "candidate")]
    offloads = [value.get("gpu_offloaded_layers") for value in placements]
    return {"baseline_present": "baseline" in models, "comparison_measured": complete,
            "improvement_measured": complete, "improvement_measured_meaning": "matched comparison, not a claim of positive improvement",
            "paired_items": len(pairs), "candidate_wins": wins, "candidate_losses": losses,
            "both_correct": sum(item["baseline_correct"] and item["candidate_correct"] for item in pairs),
            "both_incorrect": sum(not item["baseline_correct"] and not item["candidate_correct"] for item in pairs),
            "observed_accuracy_delta_pp": 100 * (wins - losses) / len(pairs) if pairs else None,
            "same_recorded_gpu_offload": offloads[0] == offloads[1] if all(value is not None for value in offloads) else None,
            "throughput_scope": "Observed server configurations, including device placement; not intrinsic codec speedup",
            "generalized_math_improvement_established": False, "items": pairs}


def http_json(url: str, body=None, *, timeout: float = 30):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # Local benchmark traffic must not be routed through a notebook's HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error


class NativeServer:
    def __init__(self, binary: Path, model: Path, args, label: str, directory: Path):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        layers = args.baseline_gpu_layers if label == "baseline" else args.gpu_layers
        self.command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                        "--ctx-size", str(args.ctx_size), "--parallel", "1", "--gpu-layers", str(layers),
                        "--fit", "on", "--flash-attn", "off", "--no-context-shift", "--no-ui",
                        "--reasoning", args.thinking, "--verbosity", "4"]
        if args.threads:
            self.command.extend(["--threads", str(args.threads)])
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = directory / (label + ".server.log")
        self.process = None
        self.log = None
        self.args = args
        self.load_seconds = None

    def start(self, on_start=None):
        self.log = self.log_path.open("a", encoding="utf-8")
        self.log.write("\nBENCHMARK SERVER START " + now() + "\n")
        self.log.flush()
        start = time.monotonic()
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT)
        if on_start:
            on_start(self.process.pid)
        while time.monotonic() - start < self.args.load_timeout:
            if self.process.poll() is not None:
                raise RuntimeError("llama-server exited during load; see " + str(self.log_path))
            try:
                if self.request("/health", timeout=2).get("status") == "ok":
                    self.load_seconds = time.monotonic() - start
                    return
            except (OSError, RuntimeError, ValueError):
                pass
            time.sleep(0.25)
        raise TimeoutError("llama-server loading timed out; see " + str(self.log_path))

    def request(self, path, body=None, *, timeout=None):
        return http_json(self.base_url + path, body, timeout=timeout or self.args.request_timeout)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if self.log is not None:
            self.log.close()


class GPUMonitor:
    """Sample server PID allocations and whole-device usage without Torch."""
    def __init__(self, interval: float = 0.2):
        self.interval, self.pid = interval, None
        self.backend = None
        self.nvml = None
        self.handles = []
        self.thread = None
        self.stop_event = threading.Event()
        self.samples, self.errors = 0, 0
        self.devices = {}
        self.process_peak = None
        self.process_samples = 0
        self.other_pids = set()
        self.last_error = None
        self.start_time = time.monotonic()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            if self.handles:
                self.backend = "NVML"
        except Exception as error:
            self.last_error = str(error)
        if self.backend is None and shutil.which("nvidia-smi"):
            self.backend = "nvidia-smi"

    def set_pid(self, pid: int):
        self.pid = pid

    def _query(self):
        devices, processes = [], None
        if self.backend == "NVML":
            processes = []
            processes_supported = True
            for handle in self.handles:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                uuid = self.nvml.nvmlDeviceGetUUID(handle)
                name = self.nvml.nvmlDeviceGetName(handle)
                devices.append({"uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                                "name": name.decode() if isinstance(name, bytes) else name,
                                "total_bytes": int(memory.total), "used_bytes": int(memory.used)})
                try:
                    for item in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        used = int(item.usedGpuMemory)
                        processes.append({"pid": int(item.pid), "bytes": used if 0 <= used < 1 << 60 else None})
                except self.nvml.NVMLError:
                    processes_supported = False
            if not processes_supported:
                processes = None
        elif self.backend == "nvidia-smi":
            result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) == 4:
                    devices.append({"uuid": row[0].strip(), "name": row[1].strip(),
                                    "total_bytes": int(float(row[2]) * (1 << 20)), "used_bytes": int(float(row[3]) * (1 << 20))})
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
                processes = []
                for row in csv.reader(result.stdout.splitlines()):
                    if len(row) == 2:
                        processes.append({"pid": int(row[0]), "bytes": int(float(row[1]) * (1 << 20)) if row[1].strip().replace(".", "", 1).isdigit() else None})
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.last_error = "Process memory unavailable: " + str(error)
        return devices, processes

    def sample(self):
        if self.backend is None:
            return
        try:
            devices, processes = self._query()
            self.samples += 1
            for current in devices:
                previous = self.devices.setdefault(current["uuid"], dict(current, baseline_used_bytes=current["used_bytes"], peak_used_bytes=current["used_bytes"]))
                previous["peak_used_bytes"] = max(previous["peak_used_bytes"], current["used_bytes"])
                previous["last_used_bytes"] = current["used_bytes"]
            if processes is not None and self.pid is not None:
                target = [item["bytes"] for item in processes if item["pid"] == self.pid]
                self.other_pids.update(item["pid"] for item in processes if item["pid"] != self.pid)
                if target and all(value is not None for value in target):
                    used = sum(target)
                    self.process_peak = max(self.process_peak or 0, used)
                    self.process_samples += 1
        except Exception as error:
            self.errors += 1
            self.last_error = type(error).__name__ + ": " + str(error)

    def start(self):
        self.sample()
        if self.backend is not None:
            def loop():
                while not self.stop_event.wait(self.interval):
                    self.sample()
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()

    def close(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=8)
        self.sample()
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return {"backend": self.backend, "available": self.samples > 0, "sample_interval_seconds": self.interval,
                "samples": self.samples, "sampling_errors": self.errors,
                "server_pid_peak_bytes": self.process_peak, "server_pid_samples": self.process_samples,
                "devices": list(self.devices.values()), "other_gpu_pids_seen": sorted(self.other_pids),
                "elapsed_seconds": time.monotonic() - self.start_time, "last_error": self.last_error,
                "scope": "Includes model loading and generation; sampled peaks are lower bounds. Device usage includes other processes and driver memory."}


def artifact(path: Path, *, gguf: bool = False) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError("Expected a file, not a checkpoint directory: " + str(path))
    if gguf:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ValueError("Expected a GGUF model file: " + str(path))
    print("Hashing", path.name, flush=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_items(path: Path, selected: list[dict], fingerprint: str) -> dict:
    if not path.exists():
        return {}
    expected = {item["item_id"]: item for item in selected}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            item = expected.get(record.get("item_id"))
            if item is None or record.get("run_fingerprint") != fingerprint or record["item_id"] in records:
                raise ValueError("Incompatible or duplicate benchmark item cache: " + str(path))
            if record.get("question_sha256") != hashlib.sha256(item["question"].encode()).hexdigest() or record.get("gold") != item["gold"]:
                raise ValueError("Benchmark item identity differs from the locked test selection")
            if hashlib.sha256(record["prompt"].encode()).hexdigest() != record["prompt_sha256"]:
                raise ValueError("Cached rendered prompt hash differs")
            records[record["item_id"]] = record
    return records


def device_placement(log: str) -> dict:
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log)
    no_gpu = "no usable GPU found" in log
    return {"gpu_offloaded_layers": int(matches[-1][0]) if matches else (0 if no_gpu else None),
            "total_offloadable_layers": int(matches[-1][1]) if matches else None,
            "no_usable_gpu_reported": no_gpu,
            "model_buffer_log_lines": [line for line in log.splitlines() if "model buffer size" in line.lower()],
            "source": "native server startup log"}


def evaluate_model(label: str, identity: dict, server_identity: dict, args, selected: list[dict],
                   fingerprint: str, reference: dict, previous: dict, *, server_factory=NativeServer,
                   monitor_factory=GPUMonitor) -> dict:
    directory = args.output_dir
    item_path = directory / (label + ".items.jsonl")
    records = load_items(item_path, selected, fingerprint)
    if len(records) == len(selected):
        result = dict(previous)
        result.update(status="COMPLETE", items=[records[item["item_id"]] for item in selected],
                      resumed_items=len(records), new_items=0)
        result["summary"] = summarize(result["items"], len(selected))
        return result
    server = server_factory(Path(server_identity["path"]), Path(identity["path"]), args, label, directory)
    monitor = monitor_factory(args.vram_interval)
    result = {"status": "RUNNING", "resumed_items": len(records), "new_items": 0,
              "server_command": server.command, "server_log": str(server.log_path), "started_utc": now()}
    monitor.start()
    try:
        server.start(on_start=monitor.set_pid)
        props = server.request("/props")
        result["server_props"] = props
        result["load_seconds"] = server.load_seconds
        effective_ctx = props.get("default_generation_settings", {}).get("n_ctx", args.ctx_size)
        if not isinstance(effective_ctx, int) or effective_ctx < args.ctx_size:
            raise ValueError("Server context is smaller than the locked benchmark setting")
        for item in selected:
            index = item["item_id"]
            if index in records:
                continue
            body = {"messages": [{"role": "user", "content": INSTRUCTION + item["question"]}],
                    "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
                    "reasoning_format": "none"}
            prompt = server.request("/apply-template", body).get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Server returned no rendered chat prompt")
            tokens = server.request("/tokenize", {"content": prompt, "add_special": True, "parse_special": True}).get("tokens")
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
                raise ValueError("Server returned invalid prompt token IDs")
            if len(tokens) + args.max_tokens > min(args.ctx_size, effective_ctx):
                raise ValueError(f"Item {index}: prompt plus generation budget exceeds context; use a larger --ctx-size for BOTH models")
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            token_hash = digest_json(tokens)
            paired = reference.get(index)
            if paired and (paired["prompt_sha256"] != prompt_hash or paired["prompt_tokens_sha256"] != token_hash):
                raise ValueError(f"Item {index}: baseline/candidate rendered prompt or tokenizer differs; paired evaluation refused")
            payload = {"prompt": prompt, "temperature": 0.0, "seed": args.seed, "n_predict": args.max_tokens,
                       "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                       "repeat_last_n": 0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                       "samplers": ["temperature"], "stream": False, "cache_prompt": False,
                       "n_keep": -1, "ignore_eos": False, "return_tokens": True}
            started = time.monotonic()
            response = server.request("/completion", payload)
            elapsed = time.monotonic() - started
            if not isinstance(response.get("content"), str):
                raise ValueError("Server response has no completion text")
            if "timings" in response and not isinstance(response["timings"], dict):
                raise ValueError("Server response has invalid timing fields")
            if response.get("truncated"):
                raise ValueError("Server truncated context despite the preflight token-budget check")
            in_think = prompt.rfind("<think>") > prompt.rfind("</think>")
            answer = final_answer(response["content"], starts_in_thinking=in_think)
            record = {"run_fingerprint": fingerprint, "item_id": index, "question": item["question"],
                      "question_sha256": hashlib.sha256(item["question"].encode()).hexdigest(), "gold": item["gold"],
                      "prompt": prompt, "prompt_sha256": prompt_hash, "prompt_token_count": len(tokens),
                      "prompt_tokens_sha256": token_hash, "content": response["content"],
                      "prediction": answer["value"], "format_ok": answer["format_ok"], "extraction_method": answer["method"],
                      "correct": answer["value"] is not None and answer["value"] == item["gold"],
                      "request_seconds": elapsed, "stop_type": response.get("stop_type"),
                      "context_truncated": bool(response.get("truncated")), "timings": response.get("timings", {}),
                      "generation_settings": response.get("generation_settings", {}), "completed_utc": now()}
            with item_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            records[index] = record
            result["new_items"] += 1
            print(f"{label}: {len(records)}/{len(selected)} item={index} correct={record['correct']} format_ok={record['format_ok']}", flush=True)
        result["status"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
    finally:
        server.close()
        result["vram"] = monitor.close()
        result["completed_utc"] = now()
    result["execution_sessions"] = previous.get("execution_sessions", []) + [
        {"started_utc": result["started_utc"], "completed_utc": result["completed_utc"],
         "new_items": result["new_items"], "server_command": result["server_command"], "vram": result["vram"]}]
    if server.log_path.exists():
        log = server.log_path.read_text(errors="replace").rsplit("BENCHMARK SERVER START", 1)[-1]
        lines = log.splitlines()
        result["offload_log_lines"] = [line for line in lines if "offload" in line.lower() or "model buffer size" in line.lower()][-30:]
        result["device_placement"] = device_placement(log)
        result["execution_sessions"][-1]["device_placement"] = result["device_placement"]
    peaks = [entry["vram"].get("server_pid_peak_bytes") for entry in result["execution_sessions"]]
    if any(peak is not None for peak in peaks):
        result["server_pid_peak_bytes_all_sessions"] = max(peak for peak in peaks if peak is not None)
    result["items"] = [records[item["item_id"]] for item in selected if item["item_id"] in records]
    result["summary"] = summarize(result["items"], len(selected))
    return result


def markdown(report: dict) -> str:
    protocol = report["protocol"]
    lines = ["# Spin8 GSM8K evaluation", "", "Status: **" + report["status"] + "**", "",
             f"Protocol: {protocol['scope']}; {protocol['samples']} / {report['dataset']['total_items']} test problems, zero-shot, greedy decoding.",
             f"Generation limit: {protocol['max_tokens']} tokens; thinking: {protocol['thinking']}; seed: {protocol['seed']}.", "",
             "This is a custom evaluation protocol, not a directly comparable published leaderboard score. No test answers are included in prompts or used to update weights.", "",
             "| Model | Completed | Numeric accuracy (Wilson 95%) | Strict format accuracy | Format failures | Extract failures | Token limits | Prompt tok/s | Decode tok/s | Server VRAM peak GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(value, places=2):
        return "unavailable" if value is None else f"{value:.{places}f}"
    for label, model in report["models"].items():
        summary = model["summary"]
        accuracy = "unavailable"
        if summary["accuracy"] is not None:
            low, high = summary["accuracy_wilson95"]
            accuracy = f"{summary['accuracy']:.1%} ({low:.1%}-{high:.1%})"
        peak = model.get("server_pid_peak_bytes_all_sessions", model.get("vram", {}).get("server_pid_peak_bytes"))
        lines.append(f"| {label} | {summary['completed_items']}/{summary['requested_items']} | {accuracy} | "
                     f"{number(None if summary['strict_format_accuracy'] is None else 100 * summary['strict_format_accuracy'])}% | "
                     f"{summary['format_failures']} | {summary['extraction_failures']} | {summary['token_limit_count']} | "
                     f"{number(summary['prompt_tokens_per_second'])} | {number(summary['decode_tokens_per_second'])} | "
                     f"{number(None if peak is None else peak / (1 << 30))} |")
    paired = report["paired"]
    lines.extend(["", "## Paired comparison", ""])
    if paired["comparison_measured"]:
        lines.append(f"Candidate wins: {paired['candidate_wins']}; losses: {paired['candidate_losses']}; both correct: {paired['both_correct']}; both incorrect: {paired['both_incorrect']}. Observed sample accuracy change: {paired['observed_accuracy_delta_pp']:+.2f} percentage points.")
        lines.append("A matched comparison was measured; this alone does not establish a general improvement in mathematical ability.")
    else:
        lines.append("No complete matched baseline comparison is available. Improvement remains unmeasured.")
    for label, model in report["models"].items():
        placement = model.get("device_placement", {})
        lines.append(f"{label} GPU layer placement: {placement.get('gpu_offloaded_layers', 'unavailable')}/{placement.get('total_offloadable_layers', 'unavailable')} (from startup log).")
    lines.extend(["", "## Interpretation and reproducibility", "",
                  "- Numeric accuracy accepts the final #### line, a final numeric boxed answer, or an unambiguous numeric final line. Strict format accuracy requires the requested #### line. Unfinished reasoning is never scored as a final answer.",
                  "- Wilson intervals describe binomial score uncertainty. A 64-item screening has substantial uncertainty and is not the full 1,319-item evaluation.",
                  "- Timing rates are time-weighted server-reported rates, preserving the server's exclusion of the first generated token from decode timing. Prompt caching is disabled. Temperature zero reduces sampling variation but does not guarantee identical floating-point behavior across hardware.",
                  "- GPU peaks are sampled lower bounds. Server PID memory is separate from whole-device memory. Other GPU processes and sampling failures are recorded in JSON.",
                  "- Automatic GPU fitting may offload different numbers of layers for F16 and Spin8. The recorded server command, offload log lines and GPU measurements must accompany speed comparisons.",
                  "- The test split is held out from this workflow; prior exposure in pretrained weights or the math donor is unknown. Do not tune weight edits or selection settings against these test results.",
                  "- Model and server SHA256 values, immutable dataset revision, exact selected indices, prompts, outputs and per-item results are recorded in benchmark.json.", "",
                  "## Source files", ""])
    for role, identity in report["inputs"].items():
        lines.append(f"- {role}: `{identity['path']}`; SHA256 `{identity['sha256']}`")
    lines.extend(["", f"Dataset: `{report['dataset']['repo']}@{report['dataset']['revision']}`; SHA256 `{report['dataset']['file_sha256']}`", "", "## Primary references", ""])
    lines.extend("- " + url for url in SOURCES)
    return "\n".join(lines) + "\n"


def run_benchmark(args, *, dataset=None, server_factory=NativeServer, monitor_factory=GPUMonitor) -> dict:
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.max_tokens < 1 or args.ctx_size <= args.max_tokens or args.seed < 0 or args.vram_interval < 0.05:
        raise ValueError("Invalid token/context budget, seed or sampling interval")
    with run_lock(args.output_dir):
        items, dataset_manifest = dataset if dataset is not None else fetch_dataset(
            Path(args.dataset_dir) if args.dataset_dir else args.output_dir / "dataset", install_missing=args.install_missing)
        selected = select_items(items, len(items) if args.full else args.samples, args.seed)
        inputs = {"server": artifact(Path(args.server_bin)), "candidate": artifact(Path(args.candidate_gguf), gguf=True)}
        if args.baseline_gguf:
            inputs["baseline"] = artifact(Path(args.baseline_gguf), gguf=True)
        protocol = {"scope": "full test split" if len(selected) == len(items) else "development screening subset",
                    "samples": len(selected), "sample_indices": [item["item_id"] for item in selected],
                    "selection": "ascending SHA256(seed:item_id)", "seed": args.seed, "few_shot": 0,
                    "temperature": 0.0, "max_tokens": args.max_tokens, "thinking": args.thinking,
                    "ctx_size": args.ctx_size, "cache_prompt": False, "instruction": INSTRUCTION,
                    "gpu_layers": args.gpu_layers, "baseline_gpu_layers": args.baseline_gpu_layers,
                    "threads": args.threads, "vram_interval": args.vram_interval,
                    "request_timeout": args.request_timeout, "load_timeout": args.load_timeout,
                    "training_performed": False, "test_answers_in_prompts": False}
        settings = {"schema": SCHEMA, "inputs": inputs, "dataset": dataset_manifest,
                    "protocol": protocol, "benchmark_source_sha256": sha256(Path(__file__))}
        fingerprint = digest_json(settings)
        plan_path = args.output_dir / "benchmark-plan.json"
        report_path = args.output_dir / "benchmark.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan.get("run_fingerprint") != fingerprint:
                raise ValueError("Benchmark settings or inputs changed; use a new output directory to preserve the prior run")
        else:
            if report_path.exists() or any(args.output_dir.glob("*.items.jsonl")):
                raise ValueError("Unowned benchmark output files exist; use a new output directory")
            atomic_json(plan_path, dict(settings, run_fingerprint=fingerprint, created_utc=now()))
        previous = json.loads(report_path.read_text()) if report_path.exists() else {}
        report = dict(settings, run_fingerprint=fingerprint, status="RUNNING", completed_utc=None,
                      environment={"python": sys.version, "platform": platform.platform(),
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                      models={}, sources=SOURCES)
        for label in (["baseline", "candidate"] if "baseline" in inputs else ["candidate"]):
            reference = {item["item_id"]: item for item in report["models"].get("baseline", {}).get("items", [])}
            model = evaluate_model(label, inputs[label], inputs["server"], args, selected, fingerprint,
                                   reference if label == "candidate" else {}, previous.get("models", {}).get(label, {}),
                                   server_factory=server_factory, monitor_factory=monitor_factory)
            report["models"][label] = model
            report["paired"] = paired_results(report["models"], len(selected))
            report["status"] = "RUNNING" if model["status"] == "COMPLETE" else "FAILED"
            atomic_json(report_path, report)
            if model["status"] != "COMPLETE":
                break
        expected_models = 2 if "baseline" in inputs else 1
        if len(report["models"]) == expected_models and all(model["status"] == "COMPLETE" for model in report["models"].values()):
            report["status"] = "COMPLETE"
        report["completed_utc"] = now()
        report["paired"] = paired_results(report["models"], len(selected))
        markdown_path = args.output_dir / "benchmark.md"
        atomic_text(markdown_path, markdown(report))
        report["artifacts"] = {"benchmark_md_sha256": sha256(markdown_path)}
        atomic_json(report_path, report)
        return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--server-bin", required=True, type=Path)
    result.add_argument("--candidate-gguf", required=True, type=Path)
    result.add_argument("--baseline-gguf", type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--dataset-dir", type=Path)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--samples", type=int, default=64)
    selection.add_argument("--full", action="store_true")
    result.add_argument("--seed", type=int, default=SEED)
    result.add_argument("--max-tokens", type=int, default=2048)
    result.add_argument("--ctx-size", type=int, default=4096)
    result.add_argument("--thinking", choices=("off", "on"), default="off")
    result.add_argument("--gpu-layers", default="auto")
    result.add_argument("--baseline-gpu-layers", default="auto")
    result.add_argument("--threads", type=int, default=0)
    result.add_argument("--load-timeout", type=float, default=900)
    result.add_argument("--request-timeout", type=float, default=1200)
    result.add_argument("--vram-interval", type=float, default=0.2)
    result.add_argument("--install-missing", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    report = run_benchmark(args)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "benchmark.json"),
                      "paired": {key: value for key, value in report["paired"].items() if key != "items"}}, indent=2))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
