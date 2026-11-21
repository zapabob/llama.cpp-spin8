#!/usr/bin/env python3
"""Exercise the built server and both completion APIs with a synthetic byte model."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def request(base: str, route: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept-Encoding": "gzip"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + route, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
        return response.status, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ngl", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(args.binary.resolve()), "-m", str(args.model.resolve()), "--host", "127.0.0.1",
               "--port", str(port), "-ngl", str(args.ngl), "-c", "128", "-b", "32", "-ub", "32",
               "-t", "2", "-np", "1", "--no-warmup", "--flash-attn", "off"]
    report = {"status": "FAIL", "scope": "Synthetic byte-tokenizer model; API correctness, not language quality",
              "binary": file_info(args.binary), "model": file_info(args.model), "command": command}
    log_path = args.out_dir / "server.log"
    report_path = args.out_dir / "server-test.json"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
                try:
                    _, health = request(base, "/health")
                    if json.loads(health).get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("server health deadline exceeded")
                time.sleep(0.2)
            report["health"] = json.loads(health)
            status, ui = request(base, "/")
            report["web_ui"] = {"http_status": status, "response_bytes": len(ui)}
            _, models = request(base, "/v1/models")
            report["models"] = json.loads(models)
            _, native = request(base, "/completion", {"prompt": [65, 66, 67], "n_predict": 3,
                                                       "temperature": 0, "seed": 42, "return_tokens": True})
            report["native_completion"] = json.loads(native)
            if report["native_completion"].get("tokens_predicted", 0) <= 0:
                raise RuntimeError("native API produced no tokens")
            _, openai = request(base, "/v1/completions", {"model": "test", "prompt": "Hi", "max_tokens": 3,
                                                         "temperature": 0, "seed": 42})
            report["openai_completion"] = json.loads(openai)
            if not report["openai_completion"].get("choices"):
                raise RuntimeError("OpenAI-compatible API returned no choice")
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
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(report_path)}))


if __name__ == "__main__":
    main()
