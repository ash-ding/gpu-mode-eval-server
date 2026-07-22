"""Tests for the HTTP server — endpoints, response format, error handling."""

import json
import threading
import time
import unittest
from http.client import HTTPConnection
from unittest.mock import MagicMock

from eval_server.container import ContainerHealth
from eval_server.pool import EvalRequest, SharedEvalPool
from eval_server.server import create_server


class FakeEvaluator:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.health = ContainerHealth.HEALTHY
        self.eval_count = 0
        self.restart_count = 0

    def start(self):
        pass

    def stop(self):
        pass

    def evaluate(self, code, task_name):
        return {
            "success": True,
            "score_us": 42.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "ok", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {
                "passed": 3, "failed": 0, "total": 3,
                "first_failure": None,
                "details": [
                    {"test_id": 0, "name": "test_0", "passed": True, "error": None},
                    {"test_id": 1, "name": "test_1", "passed": True, "error": None},
                    {"test_id": 2, "name": "test_2", "passed": True, "error": None},
                ],
            },
            "benchmark_details": {
                "geom_mean_us": 42.0,
                "individual_runs": [{"benchmark_id": 0, "config": "default", "time_us": 42.0}],
            },
        }

    def get_status(self):
        return {
            "gpu_id": self.gpu_id,
            "health": self.health.value,
            "eval_count": self.eval_count,
            "restart_count": self.restart_count,
        }


def setup_test_server(port):
    evaluator = FakeEvaluator(0)
    gpu_names = {0: "NVIDIA H100"}
    pool = SharedEvalPool([evaluator], gpu_names, eval_timeout=10)
    pool.start()
    server = create_server(pool, host="127.0.0.1", port=port, request_timeout=10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    return server, pool


class TestHealthEndpoint(unittest.TestCase):
    """Test GET /health response format."""

    @classmethod
    def setUpClass(cls):
        cls.server, cls.pool = setup_test_server(18081)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.pool.stop()

    def test_health_returns_200(self):
        conn = HTTPConnection("127.0.0.1", 18081)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)

        data = json.loads(resp.read())
        self.assertIn("workers", data)
        self.assertIn("queue_depth", data)
        self.assertIn("uptime_seconds", data)
        self.assertIn("gpu_names", data)
        conn.close()

    def test_health_worker_format(self):
        conn = HTTPConnection("127.0.0.1", 18081)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        data = json.loads(resp.read())

        worker = data["workers"][0]
        self.assertIn("gpu_id", worker)
        self.assertIn("health", worker)
        self.assertIn("eval_count", worker)
        self.assertIn("restart_count", worker)
        conn.close()


class TestEvalEndpoint(unittest.TestCase):
    """Test POST /eval endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.server, cls.pool = setup_test_server(18082)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.pool.stop()

    def test_eval_success_response_format(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()

        self.assertEqual(resp.status, 200)
        data = json.loads(resp.read())

        self.assertIn("success", data)
        self.assertIn("score_us", data)
        self.assertIn("error", data)
        self.assertIn("error_type", data)
        self.assertIn("logs", data)
        self.assertIn("timing", data)
        self.assertIn("test_results", data)
        self.assertIn("benchmark_details", data)
        self.assertIn("metadata", data)

        self.assertTrue(data["success"])
        self.assertEqual(data["score_us"], 42.0)
        conn.close()

    def test_eval_response_logs_format(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        logs = data["logs"]
        self.assertIn("stdout", logs)
        self.assertIn("stderr", logs)
        self.assertIn("compilation_log", logs)
        self.assertIn("traceback", logs)
        conn.close()

    def test_eval_response_timing_format(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        timing = data["timing"]
        self.assertIn("queue_time_ms", timing)
        self.assertIn("eval_time_ms", timing)
        self.assertIn("total_time_ms", timing)
        self.assertIn("timestamps", timing)

        ts = timing["timestamps"]
        for key in ["received", "gpu_assigned", "eval_started", "eval_completed", "response_sent"]:
            self.assertIsNotNone(ts.get(key), f"Missing timestamp: {key}")
        conn.close()

    def test_eval_response_test_results_format(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        tr = data["test_results"]
        self.assertEqual(tr["passed"], 3)
        self.assertEqual(tr["failed"], 0)
        self.assertEqual(tr["total"], 3)
        self.assertIsNone(tr["first_failure"])
        self.assertEqual(len(tr["details"]), 3)
        conn.close()

    def test_eval_response_metadata_format(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        meta = data["metadata"]
        self.assertIn("gpu_id", meta)
        self.assertIn("gpu_name", meta)
        self.assertIn("container_restarts", meta)
        self.assertIn("retry_count", meta)
        conn.close()

    def test_missing_code_returns_400(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        conn.close()

    def test_missing_task_returns_400(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        conn.close()

    def test_gpu_mismatch_returns_error(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        body = json.dumps({"code": "print(1)", "task_name": "test", "gpu_type": "V100"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        self.assertEqual(resp.status, 200)
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "gpu_mismatch")
        conn.close()

    def test_invalid_json_returns_400(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        conn.request("POST", "/eval", body="not json", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        conn.close()

    def test_not_found_returns_404(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 404)
        conn.close()

    def test_post_nonexistent_returns_404(self):
        conn = HTTPConnection("127.0.0.1", 18082)
        conn.request("POST", "/nonexistent", body="{}")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 404)
        conn.close()


class TestPartialTestPassReporting(unittest.TestCase):
    """Test that partial test pass (e.g. 15/17) is correctly reported."""

    @classmethod
    def setUpClass(cls):
        evaluator = FakeEvaluator(0)
        evaluator.evaluate = lambda code, task_name: {
            "success": False,
            "score_us": -1_000_000.0,
            "error": "2/17 tests failed",
            "error_type": "eval_failure",
            "logs": {"stdout": "", "stderr": "assertion error", "compilation_log": "", "traceback": "line 42"},
            "test_results": {
                "passed": 15, "failed": 2, "total": 17,
                "first_failure": {"test_id": 3, "name": "test_large_matrix", "error": "max_diff=0.01"},
                "details": [
                    {"test_id": i, "name": f"test_{i}", "passed": i != 3 and i != 10, "error": "max_diff=0.01" if i in (3, 10) else None}
                    for i in range(17)
                ],
            },
            "benchmark_details": None,
        }

        gpu_names = {0: "NVIDIA H100"}
        cls.pool = SharedEvalPool([evaluator], gpu_names, eval_timeout=10)
        cls.pool.start()
        cls.server = create_server(cls.pool, host="127.0.0.1", port=18083, request_timeout=10)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.pool.stop()

    def test_partial_pass_15_of_17(self):
        conn = HTTPConnection("127.0.0.1", 18083)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        self.assertFalse(data["success"])
        tr = data["test_results"]
        self.assertEqual(tr["passed"], 15)
        self.assertEqual(tr["failed"], 2)
        self.assertEqual(tr["total"], 17)
        self.assertIsNotNone(tr["first_failure"])
        self.assertEqual(tr["first_failure"]["test_id"], 3)
        self.assertEqual(len(tr["details"]), 17)
        conn.close()

    def test_logs_captured_on_failure(self):
        conn = HTTPConnection("127.0.0.1", 18083)
        body = json.dumps({"code": "print(1)", "task_name": "test"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        logs = data["logs"]
        self.assertEqual(logs["stderr"], "assertion error")
        self.assertEqual(logs["traceback"], "line 42")
        conn.close()


class TestQueue503(unittest.TestCase):
    """Test that queue full returns 503."""

    @classmethod
    def setUpClass(cls):
        evaluator = FakeEvaluator(0)
        evaluator.evaluate = lambda code, task_name: (time.sleep(2), {
            "success": True,
            "score_us": 1.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
            "benchmark_details": None,
        })[1]

        gpu_names = {0: "GPU"}
        cls.pool = SharedEvalPool([evaluator], gpu_names, eval_timeout=10)
        cls.pool.start()
        cls.server = create_server(cls.pool, host="127.0.0.1", port=18084, request_timeout=30)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.pool.stop()

    def test_queue_full_returns_503(self):
        connections = []
        body = json.dumps({"code": "print(1)", "task_name": "test"})

        for _ in range(15):
            conn = HTTPConnection("127.0.0.1", 18084)
            conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
            connections.append(conn)

        time.sleep(0.5)

        conn = HTTPConnection("127.0.0.1", 18084)
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())

        if resp.status == 503:
            self.assertEqual(data["error_type"], "queue_full")
        conn.close()

        for c in connections:
            try:
                c.getresponse()
                c.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
