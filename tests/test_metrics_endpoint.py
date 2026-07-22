"""Tests for the /metrics HTTP endpoint."""

import json
import threading
import time
import unittest
from http.client import HTTPConnection

from eval_server.container import ContainerHealth
from eval_server.pool import SharedEvalPool
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

    def evaluate(self, code, task_name, same_container_retry=0):
        return {
            "success": True,
            "score_us": 42.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
            "benchmark_details": None,
        }, None

    def get_status(self):
        return {
            "gpu_id": self.gpu_id,
            "health": self.health.value,
            "eval_count": self.eval_count,
            "restart_count": self.restart_count,
        }


class TestMetricsEndpoint(unittest.TestCase):
    """Test GET /metrics returns Prometheus-format metrics."""

    @classmethod
    def setUpClass(cls):
        evaluator = FakeEvaluator(0)
        gpu_names = {0: "NVIDIA H100"}
        cls.pool = SharedEvalPool([evaluator], gpu_names, eval_timeout=10)
        cls.pool.start()
        cls.server = create_server(cls.pool, host="127.0.0.1", port=18110, request_timeout=10)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.pool.stop()

    def test_metrics_returns_200(self):
        conn = HTTPConnection("127.0.0.1", 18110)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        body = resp.read().decode()
        self.assertIn("queue_depth", body)
        conn.close()

    def test_metrics_content_type(self):
        conn = HTTPConnection("127.0.0.1", 18110)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        resp.read()
        content_type = resp.getheader("Content-Type")
        self.assertIn("text/plain", content_type)
        conn.close()

    def test_metrics_after_eval(self):
        conn = HTTPConnection("127.0.0.1", 18110)
        body = json.dumps({"code": "print(1)", "task_name": "trimul"})
        conn.request("POST", "/eval", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()

        conn = HTTPConnection("127.0.0.1", 18110)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        metrics_body = resp.read().decode()
        self.assertIn("evals_total", metrics_body)
        conn.close()


if __name__ == "__main__":
    unittest.main()
