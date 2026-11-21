#!/usr/bin/env python3
"""Fast benchmark protocol tests; optional public-data and native-server smoke checks."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import benchmark as bench


def synthetic_dataset(count=4):
    gold = ["42", "7", "5", "12"]
    return ([{"item_id": index, "question": f"Synthetic item {index}", "gold": gold[index]}
             for index in range(count)],
            {"repo": "synthetic/protocol-test", "revision": "fixed-test-v1", "file_sha256": "synthetic",
             "total_items": count, "training_split_used": False})


class NoGPU:
    def __init__(self, interval):
        self.interval = interval
    def start(self):
        pass
    def set_pid(self, pid):
        pass
    def close(self):
        return {"backend": None, "available": False, "server_pid_peak_bytes": None, "samples": 0}


class MockServer:
    starts = []
    requests = []
    active = 0
    mismatch = False
    bad_response = False

    def __init__(self, binary, model, args, label, directory):
        self.label = label
        self.args = args
        self.command = [str(binary), "--model", str(model)]
        self.log_path = directory / (label + ".server.log")
        self.load_seconds = 0.01
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.reply({"default_generation_settings": {"n_ctx": 4096}} if self.path == "/props" else {"status": "ok"})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                MockServer.requests.append((outer.label, self.path, body))
                if self.path == "/apply-template":
                    prompt = "<user>" + body["messages"][0]["content"] + "</user><assistant>"
                    if MockServer.mismatch and outer.label == "candidate":
                        prompt += "DIFFERENT"
                    self.reply({"prompt": prompt})
                elif self.path == "/tokenize":
                    self.reply({"tokens": list(body["content"].encode())})
                elif self.path == "/completion":
                    index = int(body["prompt"].split("Synthetic item ")[1].split("<", 1)[0])
                    outputs = {"baseline": ["#### 42", "7", "#### 6", "<think>#### 12"],
                               "candidate": ["#### 43", "#### 7", r"\boxed{5}", "#### 12"]}
                    self.reply({"content": 123} if MockServer.bad_response else
                               {"content": outputs[outer.label][index], "stop_type": "eos", "truncated": False,
                                "timings": {"prompt_n": 10, "prompt_ms": 5.0, "prompt_per_second": 2000.0,
                                            "predicted_n": 4, "predicted_ms": 20.0, "predicted_per_second": 150.0, "cache_n": 0},
                                "generation_settings": {"temperature": body["temperature"], "n_predict": body["n_predict"]}})
                else:
                    self.send_error(404)
            def reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = "http://127.0.0.1:" + str(self.httpd.server_port)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def start(self, on_start=None):
        assert MockServer.active == 0, "Servers overlapped"
        MockServer.active += 1
        MockServer.starts.append(self.label)
        self.thread.start()
        if on_start:
            on_start(123)

    def request(self, path, body=None, timeout=None):
        return bench.http_json(self.base_url + path, body, timeout=3)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)
        MockServer.active -= 1


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="benchmark-test-")
        self.work = Path(self.temp.name)
        self.server = self.work / "server"
        self.server.write_bytes(b"mock-server")
        self.candidate = self.work / "candidate.gguf"
        self.baseline = self.work / "baseline.gguf"
        self.candidate.write_bytes(b"GGUFcandidate-synthetic")
        self.baseline.write_bytes(b"GGUFbaseline-synthetic")
        MockServer.starts, MockServer.requests, MockServer.active = [], [], 0
        MockServer.mismatch = MockServer.bad_response = False

    def tearDown(self):
        self.temp.cleanup()

    def args(self, *, paired=True, **changes):
        command = ["--server-bin", str(self.server), "--candidate-gguf", str(self.candidate),
                   "--output-dir", str(self.work / "results"), "--samples", "4", "--max-tokens", "8", "--ctx-size", "1024"]
        if paired:
            command.extend(["--baseline-gguf", str(self.baseline)])
        args = bench.parser().parse_args(command)
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def run_mock(self, args):
        return bench.run_benchmark(args, dataset=synthetic_dataset(), server_factory=MockServer, monitor_factory=NoGPU)

    def test_final_number_and_separate_format_status(self):
        for text, number, strict in (("Calculation\n#### 1,234.00", "1234", True),
                                     ("#### -0", "0", True), ("#### .25", "0.25", True),
                                     (r"Thus \boxed{42}", "42", False), ("The answer is 42.", "42", False),
                                     ("Answer: $7", "7", False)):
            with self.subTest(text=text):
                result = bench.final_answer(text)
                self.assertEqual(result["value"], bench.numeric_value(number))
                self.assertEqual(result["format_ok"], strict)
        for text in ("I used 42 apples and 3 bags.", "#### 1/2", "#### nan", "#### 1,23", "#### 1e999999", "No answer"):
            self.assertIsNone(bench.final_answer(text)["value"], text)

    def test_unfinished_reasoning_is_not_an_answer(self):
        self.assertIsNone(bench.final_answer("#### 42", starts_in_thinking=True)["value"])
        self.assertIsNone(bench.final_answer("<think>\n#### 42")["value"])
        self.assertEqual(bench.final_answer("#### 5</think>\n#### 42", starts_in_thinking=True)["value"], "42")
        self.assertEqual(bench.final_answer("<think>#### 5</think>\n#### 42")["value"], "42")

    def test_sampling_is_deterministic_nested_and_unique(self):
        items, _ = synthetic_dataset()
        short = bench.select_items(items, 2, 1)
        full = bench.select_items(list(reversed(items)), 4, 1)
        self.assertEqual(short, full[:2])
        self.assertEqual(len({item["item_id"] for item in full}), 4)
        with self.assertRaises(ValueError):
            bench.select_items(items, 5, 1)

    def test_wilson_interval_known_limits(self):
        zero = bench.wilson_interval(0, 64)
        half = bench.wilson_interval(32, 64)
        one = bench.wilson_interval(64, 64)
        self.assertAlmostEqual(zero[0], 0, places=12)
        self.assertAlmostEqual(zero[1], 0.05662405979280535, places=12)
        self.assertAlmostEqual(half[0] + half[1], 1, places=12)
        self.assertAlmostEqual(one[1], 1, places=12)
        self.assertIsNone(bench.wilson_interval(0, 0))

    def test_paired_http_evaluation_and_provenance(self):
        args = self.args()
        report = self.run_mock(args)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(MockServer.starts, ["baseline", "candidate"])
        self.assertEqual(MockServer.active, 0)
        self.assertEqual(report["models"]["baseline"]["summary"]["accuracy"], 0.5)
        candidate = report["models"]["candidate"]["summary"]
        self.assertEqual(candidate["accuracy"], 0.75)
        self.assertEqual(candidate["format_failures"], 1)
        self.assertEqual(candidate["strict_format_accuracy"], 0.5)
        self.assertEqual(candidate["prompt_tokens_per_second"], 2000)
        self.assertEqual(candidate["decode_tokens_per_second"], 150)
        self.assertEqual(report["paired"]["candidate_wins"], 2)
        self.assertEqual(report["paired"]["candidate_losses"], 1)
        self.assertEqual(report["paired"]["observed_accuracy_delta_pp"], 25)
        self.assertTrue(report["paired"]["comparison_measured"])
        self.assertFalse(report["paired"]["generalized_math_improvement_established"])
        self.assertEqual(report["inputs"]["candidate"]["sha256"], bench.sha256(self.candidate))
        self.assertEqual(report["inputs"]["server"]["sha256"], bench.sha256(self.server))
        self.assertEqual(report["artifacts"]["benchmark_md_sha256"], bench.sha256(args.output_dir / "benchmark.md"))
        completions = [body for _, path, body in MockServer.requests if path == "/completion"]
        self.assertEqual(len(completions), 8)
        for body in completions:
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["n_predict"], 8)
            self.assertFalse(body["cache_prompt"])
        self.assertEqual([body["prompt"] for body in completions[:4]], [body["prompt"] for body in completions[4:]])

    def test_candidate_only_does_not_claim_comparison(self):
        report = self.run_mock(self.args(paired=False))
        self.assertEqual(report["status"], "COMPLETE")
        self.assertFalse(report["paired"]["baseline_present"])
        self.assertFalse(report["paired"]["improvement_measured"])
        self.assertIsNone(report["paired"]["observed_accuracy_delta_pp"])
        self.assertIsNone(report["models"]["candidate"]["vram"]["server_pid_peak_bytes"])

    def test_prompt_mismatch_stops_candidate_before_generation(self):
        MockServer.mismatch = True
        report = self.run_mock(self.args())
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("tokenizer differs", report["models"]["candidate"]["error"])
        self.assertFalse(report["paired"]["comparison_measured"])
        self.assertFalse(any(label == "candidate" and path == "/completion" for label, path, _ in MockServer.requests))
        self.assertEqual(MockServer.active, 0)

    def test_insufficient_context_fails_before_generation(self):
        report = self.run_mock(self.args(paired=False, ctx_size=64, max_tokens=8))
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("exceeds context", report["models"]["candidate"]["error"])
        self.assertFalse(any(path == "/completion" for _, path, _ in MockServer.requests))

    def test_resume_reuses_results_and_changed_settings_rejected(self):
        args = self.args()
        first = self.run_mock(args)
        second = self.run_mock(args)
        self.assertEqual(MockServer.starts, ["baseline", "candidate"])
        self.assertEqual(second["models"]["candidate"]["new_items"], 0)
        self.assertEqual(first["models"]["candidate"]["items"], second["models"]["candidate"]["items"])
        args.max_tokens += 1
        with self.assertRaisesRegex(ValueError, "settings or inputs changed"):
            self.run_mock(args)

    def test_directory_and_non_gguf_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            self.run_mock(self.args(paired=False, candidate_gguf=self.work))
        self.candidate.write_bytes(b"not-gguf")
        with self.assertRaisesRegex(ValueError, "GGUF model file"):
            self.run_mock(self.args(paired=False))

    def test_invalid_endpoint_response_cleans_up_and_is_failed(self):
        MockServer.bad_response = True
        report = self.run_mock(self.args(paired=False))
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["models"]["candidate"]["summary"]["completed_items"], 0)
        self.assertFalse(report["paired"]["comparison_measured"])
        self.assertEqual(MockServer.active, 0)

    def test_gpu_process_peak_excludes_other_processes(self):
        monitor = bench.GPUMonitor.__new__(bench.GPUMonitor)
        monitor.backend, monitor.pid = "synthetic", 10
        monitor.samples = monitor.errors = monitor.process_samples = 0
        monitor.devices, monitor.other_pids = {}, set()
        monitor.process_peak, monitor.last_error = None, None
        snapshots = iter([
            ([{"uuid": "GPU-test", "name": "Synthetic", "total_bytes": 1000, "used_bytes": 700}],
             [{"pid": 10, "bytes": 200}, {"pid": 20, "bytes": 400}]),
            ([{"uuid": "GPU-test", "name": "Synthetic", "total_bytes": 1000, "used_bytes": 900}],
             [{"pid": 10, "bytes": 500}, {"pid": 20, "bytes": 300}]),
            ([{"uuid": "GPU-test", "name": "Synthetic", "total_bytes": 1000, "used_bytes": 900}],
             [{"pid": 10, "bytes": None}]),
        ])
        monitor._query = lambda: next(snapshots)
        for _ in range(3):
            monitor.sample()
        self.assertEqual(monitor.process_peak, 500)
        self.assertEqual(monitor.devices["GPU-test"]["peak_used_bytes"], 900)
        self.assertEqual(monitor.other_pids, {20})
        self.assertEqual(monitor.process_samples, 2)

    def test_recorded_offload_is_actual_startup_value(self):
        result = bench.device_placement("load_tensors: offloaded 37/65 layers to GPU\nload_tensors: CUDA0 model buffer size = 200.0 MiB")
        self.assertEqual(result["gpu_offloaded_layers"], 37)
        self.assertEqual(result["total_offloadable_layers"], 65)
        self.assertEqual(len(result["model_buffer_log_lines"]), 1)
        self.assertEqual(bench.device_placement("no usable GPU found")["gpu_offloaded_layers"], 0)
        self.assertIsNone(bench.device_placement("")["gpu_offloaded_layers"])


def native_smoke(binary: Path, model: Path, directory: Path) -> dict:
    args = bench.parser().parse_args(["--server-bin", str(binary), "--candidate-gguf", str(model),
        "--output-dir", str(directory), "--samples", "2", "--max-tokens", "8", "--ctx-size", "128",
        "--gpu-layers", "0", "--load-timeout", "45", "--request-timeout", "45", "--threads", "2"])
    with patch.object(bench, "INSTRUCTION", "Answer with #### <number>.\n"):
        report = bench.run_benchmark(args, dataset=synthetic_dataset(2))
    if report["status"] != "COMPLETE":
        raise AssertionError(report["models"]["candidate"].get("error"))
    return {"status": "PASS", "completed_requests": report["models"]["candidate"]["summary"]["completed_items"],
            "server_sha256": report["inputs"]["server"]["sha256"], "model_sha256": report["inputs"]["candidate"]["sha256"],
            "scope": "Native HTTP/template/tokenizer/timing integration with synthetic questions and fixture weights; not a math-quality result",
            "cuda_executed": False, "report": str(directory / "benchmark.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--live-dataset-dir", type=Path)
    parser.add_argument("--native-server", type=Path)
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--native-output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BenchmarkTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "real_model_quality_measured": False,
              "scope": "Synthetic HTTP protocol, scoring, uncertainty, paired comparison, caching and memory-sampling accounting"}
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.live_dataset_dir:
        items, manifest = bench.fetch_dataset(args.live_dataset_dir)
        chosen = bench.select_items(items, 64, bench.SEED)
        report["live_dataset"] = dict(manifest, parsed_gold_answers=len(items), samples=len(chosen),
                                      selected_ids_sha256=bench.digest_json([item["item_id"] for item in chosen]))
    if args.native_server or args.native_model:
        if not args.native_server or not args.native_model or not args.native_output:
            parser.error("Native smoke requires --native-server, --native-model and --native-output")
        report["native_cpu_smoke"] = native_smoke(args.native_server, args.native_model, args.native_output)
    if args.report:
        bench.atomic_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
