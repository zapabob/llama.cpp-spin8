#!/usr/bin/env python3
"""Build the pinned Spin8 runtime without replacing Colab's CUDA/PyTorch stack.

All native correctness checks are mandatory. PASS covers the supplied synthetic
fixtures, not 27B model quality or speed. The default server exposes its APIs;
its optional web UI is excluded to avoid the shallow-clone b1/latest download.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

BASE_COMMIT = "7fce5439fe451cd48340b112ec37d2befe02c776"
SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
REQUIRED = {
    "numpy": "numpy>=2,<3", "scipy.linalg": "scipy>=1.11,<2",
    "safetensors": "safetensors>=0.5,<1", "huggingface_hub": "huggingface-hub>=1,<2",
    "transformers": "transformers>=5,<6", "sentencepiece": "sentencepiece>=0.2,<0.3",
    "google.protobuf": "protobuf>=4.21", "yaml": "PyYAML>=6,<7",
    "tqdm": "tqdm>=4.66,<5", "packaging": "packaging>=24",
}
FIXTURES = (
    "tiny-qwen35-spin8.gguf", "tiny-qwen35-decoded.gguf",
    "tiny-qwen35-projected-spin8.gguf", "tiny-qwen35-projected-control.gguf",
)

PROBE_CODE = r'''
import importlib, importlib.metadata as md, json, platform, sys, traceback
from pathlib import Path
out = {"python": sys.version, "python_executable": sys.executable,
       "platform": platform.platform(), "modules": {}, "distributions": {}}
for dist in md.distributions():
    name = dist.metadata.get("Name")
    if name:
        out["distributions"][name] = dist.version
for name in json.loads(sys.argv[2]):
    try:
        module = importlib.import_module(name)
        if name == "transformers":
            from transformers import AutoConfig
        out["modules"][name] = {"status": "ok", "version": getattr(module, "__version__", None)}
    except ModuleNotFoundError as exc:
        missing = exc.name == name or name.startswith(str(exc.name) + ".")
        out["modules"][name] = {"status": "missing" if missing else "broken", "error": str(exc)}
    except Exception as exc:
        out["modules"][name] = {"status": "broken", "error": repr(exc)}
try:
    import torch
    out["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available()}
    if out["modules"].get("numpy", {}).get("status") == "ok":
        import numpy as np
        assert torch.from_numpy(np.array([1.0], dtype=np.float32)).add(1).item() == 2.0
    out["torch"]["numpy_bridge"] = out["modules"].get("numpy", {}).get("status") == "ok"
    if sys.argv[3] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. In Colab choose an A100 GPU runtime and reconnect.")
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        result = torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda")
        torch.cuda.synchronize()
        assert result[0, 0].item() == 8.0
        out["gpu"] = {"name": props.name, "index": device,
                      "capability": list(torch.cuda.get_device_capability(device)),
                      "total_memory_bytes": props.total_memory, "torch_device_executed": True}
        maps = Path("/proc/self/maps")
        out["loaded_cudart"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                      if "/libcudart.so" in line}) if maps.exists() else []
except Exception as exc:
    out["torch_error"] = repr(exc)
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
'''


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def capture(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    return result.stdout.strip()


def run(command: list[str], *, log: Path, cwd: Path | None = None,
        env: dict | None = None, heartbeat: float = 20.0,
        console_limit: int | None = None) -> None:
    """Tee child output and print a heartbeat during silent compiler steps."""
    display = [command[0], "-c", "<fresh Python environment probe>"] if len(command) > 2 and command[1] == "-c" else command
    print("Running:", shlex.join(display), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    messages: queue.Queue = queue.Queue()
    started = time.monotonic()
    line_count = 0
    tail = []
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        def reader() -> None:
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            while True:
                try:
                    line = messages.get(timeout=heartbeat)
                except queue.Empty:
                    print(f"Still running ({time.monotonic() - started:.0f}s); log: {log}", flush=True)
                    continue
                if line is None:
                    break
                handle.write(line)
                handle.flush()
                line_count += 1
                tail = [*tail[-19:], line]
                if console_limit is None or line_count <= console_limit:
                    print(line, end="", flush=True)
                elif line_count == console_limit + 1:
                    print(f"Further native details are saved to {log}", flush=True)
            code = process.wait()
            if code:
                if console_limit is not None and line_count > console_limit:
                    print("".join(tail), end="", flush=True)
                raise RuntimeError(f"Command exited {code}: {shlex.join(command)}; full log: {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            process.stdout.close()
            thread.join(timeout=2)


def safe_payload_file(payload: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name
            or str(relative) != name):
        raise ValueError("Unsafe payload path: " + name)
    path = payload / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(payload.resolve()):
        raise ValueError("Payload file is missing or not a regular contained file: " + name)
    return path


def validate_payload(payload: Path) -> dict:
    manifest = json.loads(safe_payload_file(payload, "MANIFEST.json").read_text())
    if manifest.get("base_commit") != BASE_COMMIT or manifest.get("upstream", SOURCE_URL) != SOURCE_URL:
        raise ValueError("Payload was produced for a different upstream revision")
    patch = safe_payload_file(payload, "spin8-llama.patch")
    if sha256(patch) != manifest.get("patch_sha256"):
        raise ValueError("Source patch SHA-256 does not match MANIFEST.json")
    seen = set()
    for entry in manifest.get("fixture_files", []):
        name = entry["path"]
        if name in seen:
            raise ValueError("Duplicate fixture in manifest: " + name)
        seen.add(name)
        path = safe_payload_file(payload, name)
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Fixture size or SHA-256 mismatch: " + name)
    if not {"fixtures/" + name for name in FIXTURES}.issubset(seen):
        raise ValueError("Payload omits a mandatory native or projected parity fixture")
    return manifest


def fresh_probe(report: Path, device: str, log: Path) -> dict:
    run([sys.executable, "-c", PROBE_CODE, str(report), json.dumps(["torch", *REQUIRED]), device], log=log)
    return json.loads(report.read_text())


def dependency_plan(probe: dict) -> list[str]:
    torch_state = probe["modules"].get("torch", {})
    if torch_state.get("status") != "ok" or probe.get("torch_error"):
        raise RuntimeError("The runtime's existing PyTorch is not usable. "
                           "Reconnect a current Colab runtime; this installer never replaces PyTorch. "
                           + str(probe.get("torch_error", torch_state)))
    broken = {name: state.get("error") for name, state in probe["modules"].items()
              if state.get("status") == "broken"}
    if broken:
        raise RuntimeError("Installed imports are broken; reconnect a clean runtime: " + json.dumps(broken))
    return [requirement for name, requirement in REQUIRED.items()
            if probe["modules"].get(name, {}).get("status") != "ok"]


def cuda_architecture(probe: dict) -> str:
    gpu = probe.get("gpu", {})
    capability = gpu.get("capability")
    if gpu.get("torch_device_executed") is not True or not isinstance(capability, list) or len(capability) != 2:
        raise RuntimeError("An actual CUDA tensor operation has not succeeded")
    if any(type(value) is not int or value < 0 for value in capability) or capability[0] < 7:
        raise RuntimeError("Unsupported CUDA capability: " + repr(capability))
    return str(capability[0]) + str(capability[1])


def prepare_python(reports: Path, device: str) -> dict:
    before = fresh_probe(reports / "environment-before.json", device, reports / "probe-before.log")
    missing = dependency_plan(before)
    if missing:
        # Freeze every already-installed distribution. This prevents transitive
        # dependency resolution from downgrading torch/CUDA/numpy in Colab.
        constraints = reports / "installed-constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in
                                       sorted(before["distributions"].items())
                                       if re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                                       and "\n" not in version), encoding="utf-8")
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--constraint", str(constraints), *missing], log=reports / "dependencies.log")
    after = fresh_probe(reports / "environment.json", device, reports / "probe-after.log")
    if dependency_plan(after):
        raise RuntimeError("Required imports are still missing after installation")
    for name, version in before["distributions"].items():
        if after["distributions"].get(name) != version:
            raise RuntimeError(f"An existing package unexpectedly changed: {name}; {version} -> "
                               + str(after["distributions"].get(name)))
    after["installed_missing_requirements"] = missing
    after["existing_packages_preserved"] = True
    atomic_json(reports / "environment.json", after)
    return after


def find_tool(name: str, extras: tuple[Path, ...] = ()) -> str | None:
    for candidate in [Path(sys.executable).parent / name, *extras]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return shutil.which(name)


def prepare_tools(reports: Path, probe: dict, device: str) -> dict:
    tools = {}
    for name in ("git", "c++", "cc"):
        tools[name] = find_tool(name)
        if not tools[name]:
            raise RuntimeError(f"Missing {name}; use a standard current Linux/Colab runtime")
    cmake = find_tool("cmake")
    version_text = capture([cmake, "--version"]) if cmake else ""
    version_match = re.search(r"cmake version (\d+)\.(\d+)\.(\d+)", version_text)
    if not version_match or tuple(map(int, version_match.groups())) < (3, 28, 0):
        run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "cmake==3.31.10"],
            log=reports / "cmake-install.log")
        cmake = find_tool("cmake")
    if not cmake:
        raise RuntimeError("CMake was installed but its executable could not be found")
    tools["cmake"] = cmake
    tools["cmake_version"] = capture([cmake, "--version"]).splitlines()[0]
    tools["cxx_version"] = capture([tools["c++"], "--version"]).splitlines()[0]
    tools["cc_version"] = capture([tools["cc"], "--version"]).splitlines()[0]
    if device == "cuda":
        roots = [Path(os.environ[name]) for name in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(name)]
        roots.extend([Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"), reverse=True)])
        nvcc = find_tool("nvcc", tuple(root / "bin/nvcc" for root in roots))
        if not nvcc:
            raise RuntimeError("CUDA GPU is present but nvcc is missing; select a standard Colab GPU runtime")
        arch = cuda_architecture(probe)
        supported = capture([nvcc, "--list-gpu-code"])
        if "sm_" + arch not in supported.split():
            raise RuntimeError(f"This nvcc cannot generate sm_{arch}; supported GPU code:\n{supported}")
        root = Path(nvcc).resolve().parent.parent
        runtime_candidates = [root / "lib64/libcudart.so", root / "targets/x86_64-linux/lib/libcudart.so"]
        runtime_candidates.extend(Path(value) for value in probe.get("loaded_cudart", []))
        runtime = next((value.resolve() for value in runtime_candidates if value.is_file()), None)
        if not runtime:
            raise RuntimeError("Cannot locate the CUDA runtime library for mandatory native GPU checks")
        tools.update({"nvcc": nvcc, "nvcc_version": capture([nvcc, "--version"]),
                      "cuda_root": str(root), "cuda_arch": arch, "cudart": str(runtime)})
    return tools


def prepare_source(repo: Path, payload: Path, reports: Path, git: str) -> dict:
    patch = payload / "spin8-llama.patch"
    if not repo.exists():
        repo.mkdir(parents=True)
    if not (repo / ".git").is_dir():
        if any(repo.iterdir()):
            raise RuntimeError("Existing source directory is not a git checkout; use a new work directory")
        run([git, "init", str(repo)], log=reports / "git-init.log")
        run([git, "remote", "add", "origin", SOURCE_URL], cwd=repo, log=reports / "git-remote.log")
    head = subprocess.run([git, "rev-parse", "--verify", "HEAD"], cwd=repo, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    remote = subprocess.run([git, "remote", "get-url", "origin"], cwd=repo, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if remote.returncode and head.returncode and not any(path.name != ".git" for path in repo.iterdir()):
        # Resume interruption between `git init` and `remote add` safely.
        run([git, "remote", "add", "origin", SOURCE_URL], cwd=repo, log=reports / "git-remote.log")
        origin = SOURCE_URL
    else:
        origin = remote.stdout.strip()
    if origin != SOURCE_URL:
        raise RuntimeError("Source origin differs from the pinned upstream; use a new work directory")
    if head.returncode:
        if any(path.name != ".git" for path in repo.iterdir()):
            raise RuntimeError("Interrupted checkout contains files; preserving it. Use a new work directory")
        run([git, "fetch", "--depth", "1", "origin", BASE_COMMIT], cwd=repo, log=reports / "git-fetch.log")
        run([git, "checkout", "--detach", BASE_COMMIT], cwd=repo, log=reports / "git-checkout.log")
    elif head.stdout.strip() != BASE_COMMIT:
        raise RuntimeError("Existing checkout has a different revision; preserving it. Use a new work directory")
    reverse = subprocess.run([git, "apply", "--reverse", "--check", str(patch)], cwd=repo,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if reverse.returncode:
        run([git, "apply", "--check", str(patch)], cwd=repo, log=reports / "patch-check.log")
        run([git, "apply", str(patch)], cwd=repo, log=reports / "patch-apply.log")
    # Verify all patched file contents against an independent, temporary Git
    # index. Merely accepting `git apply --reverse --check` misses extra edits.
    paths = re.findall(r"^\+\+\+ b/(.+)$", patch.read_text(), flags=re.MULTILINE)
    if not paths or any(".." in PurePosixPath(name).parts for name in paths):
        raise RuntimeError("Unsupported patch path list")
    changes = set(capture([git, "diff", "--name-only", "HEAD"], cwd=repo).splitlines())
    untracked = set(capture([git, "ls-files", "--others", "--exclude-standard"], cwd=repo).splitlines())
    unexpected = (changes | untracked) - set(paths)
    if unexpected:
        raise RuntimeError("Checkout has additional source changes; preserving them: " + ", ".join(sorted(unexpected)))
    with tempfile.TemporaryDirectory(prefix="source-index-", dir=reports) as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index")}
        capture([git, "read-tree", BASE_COMMIT], cwd=repo, env=env)
        capture([git, "apply", "--cached", str(patch)], cwd=repo, env=env)
        expected = {}
        for line in capture([git, "ls-files", "--stage", "--", *paths], cwd=repo, env=env).splitlines():
            mode_hash, name = line.split("\t", 1)
            expected[name] = mode_hash.split()[1]
        actual = capture([git, "hash-object", "--", *paths], cwd=repo).splitlines()
        if len(actual) != len(paths) or any(expected.get(name) != digest for name, digest in zip(paths, actual)):
            raise RuntimeError("Patched source contents differ from the verified payload; use a new work directory")
    return {"base_commit": BASE_COMMIT, "patch_sha256": sha256(patch), "origin": origin,
            "patched_files": {name: sha256(repo / name) for name in paths}, "verified": True}


def api_request(base: str, route: str, body: dict | None = None, timeout: float = 45) -> tuple[int, bytes]:
    request = urllib.request.Request(base + route, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Accept-Encoding": "gzip"})
    # The smoke test is exclusively loopback. Ignore proxy environment settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        content = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            content = gzip.decompress(content)
        return response.status, content


def server_smoke(binary: Path, model: Path, reports: Path, device: str, with_ui: bool) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
               "-ngl", "99" if device == "cuda" else "0", "-c", "128", "-b", "32", "-ub", "32",
               "-t", "2", "-np", "1", "--no-warmup", "--flash-attn", "off",
               "--ui" if with_ui else "--no-ui"]
    report = {"status": "FAIL", "scope": "Synthetic byte model; API correctness only", "command": command,
              "web_ui_enabled": with_ui, "binary": file_info(binary), "model": file_info(model)}
    log_path = reports / "server.log"
    with log_path.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {log_path}")
                try:
                    _, content = api_request(base, "/health", timeout=2)
                    health = json.loads(content)
                    if health.get("status") == "ok":
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server health deadline exceeded; see {log_path}")
                time.sleep(0.2)
            report["health"] = health
            if with_ui:
                status, content = api_request(base, "/")
                if status != 200 or not content:
                    raise RuntimeError("Enabled web UI returned no successful content")
                report["web_ui"] = {"http_status": status, "response_bytes": len(content)}
            _, content = api_request(base, "/v1/models")
            report["models"] = json.loads(content)
            _, content = api_request(base, "/completion", {"prompt": [65, 66, 67], "n_predict": 3,
                                "temperature": 0, "seed": 42, "return_tokens": True})
            report["native_completion"] = json.loads(content)
            if report["native_completion"].get("tokens_predicted", 0) <= 0:
                raise RuntimeError("Native completion API produced no tokens")
            _, content = api_request(base, "/v1/completions", {"model": "test", "prompt": "Hi",
                                        "max_tokens": 3, "temperature": 0, "seed": 42})
            report["openai_completion"] = json.loads(content)
            if not report["openai_completion"].get("choices"):
                raise RuntimeError("OpenAI-compatible completion API produced no choices")
            report["status"] = "PASS"
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            report["server_log"] = file_info(log_path)
            atomic_json(reports / "server-test.json", report)
    print("Server API smoke: PASS", flush=True)
    return report


def require_parity(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "PASS" or value.get("comparison", {}).get("all_observed_required_op_output_devices_match") is not True:
        raise RuntimeError("Native parity or actual output device verification failed: " + str(path))
    return value


def require_cuda_execution(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("cuda_executed") is not True:
        raise RuntimeError("CUDA execution was skipped or failed: " + str(path))
    return value


@contextmanager
def work_lock(work: Path):
    import fcntl
    with (work / ".spin8-bootstrap.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another build is using this work directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(args: argparse.Namespace) -> dict:
    work = args.work_dir.expanduser().resolve()
    payload = args.payload_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    reports = work / "reports"
    reports.mkdir(exist_ok=True)
    report_path = reports / "build-report.json"
    report = {"schema": "spin8-native-build-v2", "started_utc": now(), "status": "RUNNING",
              "device_requested": args.device, "base_commit": BASE_COMMIT, "stages": [],
              "cuda_device_execution": False, "web_ui_enabled": args.with_ui,
              "scope": "Synthetic runtime correctness; no 27B quality or performance claim"}
    @contextmanager
    def stage(name: str):
        item = {"name": name, "status": "RUNNING", "started_utc": now()}
        report["stages"].append(item)
        report["current_stage"] = name
        atomic_json(report_path, report)
        print(f"\n[{len(report['stages'])}] {name}", flush=True)
        started = time.monotonic()
        try:
            yield
            item["status"] = "PASS"
        except BaseException as error:
            item["status"] = "FAIL"
            report.update({"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic_json(report_path, report)
    with work_lock(work):
        with stage("validate payload"):
            manifest = validate_payload(payload)
            report["patch_sha256"] = manifest["patch_sha256"]
        with stage("probe and preserve Python environment"):
            probe = prepare_python(reports, args.device)
            report["environment_report"] = str(reports / "environment.json")
        with stage("detect native toolchain and actual GPU architecture"):
            toolchain = prepare_tools(reports, probe, args.device)
            probe["toolchain"] = toolchain
            atomic_json(reports / "environment.json", probe)
            report["toolchain"] = toolchain
        repo = work / "llama.cpp"
        with stage("fetch pinned source and verify complete patch"):
            report["source"] = prepare_source(repo, payload, reports, toolchain["git"])
            report["source_dir"] = str(repo)
        # A distinct build directory avoids inheriting stale CUDA/UI flags.
        build = repo / ("build-spin8-v2-" + args.device + ("-ui" if args.with_ui else ""))
        report["build_dir"] = str(build)
        report["cuda_arch"] = toolchain.get("cuda_arch")
        environment = os.environ.copy()
        if args.with_ui:
            # Explicit opt-in; archive identity is recorded below. The default
            # never downloads mutable UI assets or derives the invalid b1 tag.
            environment.pop("HF_WEBUI_VERSION", None)
            environment["HF_UI_VERSION"] = args.ui_version
        configure = [toolchain["cmake"], "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     "-DGGML_NATIVE=OFF", "-DGGML_RPC=OFF", "-DLLAMA_BUILD_TESTS=ON",
                     "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON",
                     "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_UI=OFF",
                     "-DLLAMA_USE_PREBUILT_UI=" + ("ON" if args.with_ui else "OFF"),
                     "-DGGML_CUDA=" + ("ON" if args.device == "cuda" else "OFF"),
                     "-DCMAKE_C_COMPILER=" + toolchain["cc"], "-DCMAKE_CXX_COMPILER=" + toolchain["c++"]]
        if args.device == "cuda":
            configure += ["-DCMAKE_CUDA_ARCHITECTURES=" + toolchain["cuda_arch"],
                          "-DCMAKE_CUDA_COMPILER=" + toolchain["nvcc"],
                          "-DCMAKE_CUDA_HOST_COMPILER=" + toolchain["c++"],
                          "-DCUDAToolkit_ROOT=" + toolchain["cuda_root"],
                          "-DGGML_CUDA_NO_VMM=ON", "-DGGML_CUDA_FA=OFF", "-DGGML_CUDA_NCCL=OFF"]
        with stage("configure native build"):
            run(configure, log=reports / "configure.log", env=environment)
        with stage("compile native server and graph test"):
            run([toolchain["cmake"], "--build", str(build), "--target", "llama-server", "test-spin8-native",
                 "-j", str(args.jobs)], log=reports / "build.log", env=environment)
            for name in ("llama-server", "test-spin8-native"):
                report[name] = file_info(build / "bin" / name)
            report["libraries"] = {path.name: file_info(path) for path in (build / "bin").glob("*.so*")
                                    if path.is_file() and not path.is_symlink()}
            if args.with_ui:
                report["ui_requested_version"] = args.ui_version
                report["ui_archives"] = [file_info(path) for path in build.rglob("dist.tar.gz")]
        scripts = repo / "tools/spin8"
        with stage("CPU codec correctness"):
            run([sys.executable, str(scripts / "test_native_codec.py"), "--library", str(build / "bin/libggml-base.so"),
                 "--report", str(reports / "codec-cpu.json")], log=reports / "codec-cpu.log")
        if args.device == "cuda":
            with stage("CUDA codec actual device correctness"):
                run([sys.executable, str(scripts / "test_cuda_codec.py"), "--arch", "sm_" + toolchain["cuda_arch"],
                     "--nvcc", toolchain["nvcc"], "--cudart", toolchain["cudart"], "--require-device",
                     "--report", str(reports / "codec-cuda.json")], log=reports / "codec-cuda.log")
                require_cuda_execution(reports / "codec-cuda.json")
        verify = [sys.executable, str(scripts / "verify_native.py"), "--binary", str(build / "bin/test-spin8-native"),
                  "--ngl", "99" if args.device == "cuda" else "0", "--control-ngl", "0",
                  "--require-spin8-op", "MUL_MAT_SPIN8", "--require-spin8-device",
                  "CUDA" if args.device == "cuda" else "CPU"]
        for name, model, control in (("native-parity", FIXTURES[0], FIXTURES[1]),
                                      ("projected-native-parity", FIXTURES[2], FIXTURES[3])):
            with stage(name):
                run([*verify, "--model", str(payload / "fixtures" / model), "--control", str(payload / "fixtures" / control),
                     "--out-dir", str(reports / name)], log=reports / (name + ".log"), console_limit=25)
                parity = require_parity(reports / name / "parity.json")
                print(json.dumps({"status": "PASS", "check": name,
                                  "max_absolute_error": parity["comparison"]["max_absolute_error"],
                                  "actual_op_device_verified": True}), flush=True)
        with stage("server generation APIs"):
            server_smoke(build / "bin/llama-server", payload / "fixtures" / FIXTURES[0], reports / "server", args.device, args.with_ui)
        report.update({"status": "PASS", "completed_utc": now(), "cuda_device_execution": args.device == "cuda",
                       "projected_native_parity": "PASS", "current_stage": "complete"})
        atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "build_dir": str(build), "report": str(report_path),
                      "cuda_device_execution": report["cuda_device_execution"]}, indent=2), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Prepare dependencies, build and run all mandatory checks")
    command.add_argument("--work-dir", type=Path, required=True)
    command.add_argument("--payload-dir", type=Path, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    command.add_argument("--jobs", type=int, default=4)
    command.add_argument("--with-ui", action="store_true", help="Opt in to external server UI asset download")
    command.add_argument("--ui-version", default="latest", help="Requested optional UI asset version; archive identity is recorded")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        prepare(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Bootstrap failed: {error}", file=sys.stderr, flush=True)
        print(f"Inspect {args.work_dir.resolve() / 'reports/build-report.json'} and the named stage log.",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
