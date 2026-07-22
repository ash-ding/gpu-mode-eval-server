"""Tests for the Python client library."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from eval_server.client import EvalClient, EvalResult


class TestEvalResult(unittest.TestCase):
    """Test EvalResult accessor methods."""

    def test_success_result(self):
        data = {
            "success": True,
            "score_us": 12345.0,
            "error": None,
            "error_type": None,
            "logs": {"traceback": None},
            "timing": {"queue_time_ms": 120, "eval_time_ms": 530},
            "test_results": {
                "passed": 3,
                "total": 3,
                "details": [
                    {"passed": True},
                    {"passed": True},
                    {"passed": True},
                ],
            },
            "benchmark_details": {"geom_mean_us": 12345.0},
            "metadata": {"gpu_id": 0},
        }
        result = EvalResult(data)
        self.assertTrue(result.success)
        self.assertEqual(result.score_us, 12345.0)
        self.assertIsNone(result.error)
        self.assertIsNone(result.get_traceback())
        self.assertEqual(result.get_queue_time(), 120)
        self.assertEqual(result.get_eval_time(), 530)
        self.assertEqual(len(result.get_failed_tests()), 0)

    def test_failure_result_with_traceback(self):
        data = {
            "success": False,
            "score_us": -1_000_000.0,
            "error": "RuntimeError",
            "error_type": "eval_failure",
            "logs": {"traceback": "Traceback (most recent call last):\n  File ..."},
            "timing": {"queue_time_ms": 50, "eval_time_ms": 200},
            "test_results": {
                "passed": 15,
                "failed": 2,
                "total": 17,
                "details": [
                    {"passed": True},
                    {"passed": False, "error": "assertion failed"},
                ],
            },
        }
        result = EvalResult(data)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "eval_failure")
        self.assertIn("Traceback", result.get_traceback())
        self.assertEqual(len(result.get_failed_tests()), 1)

    def test_default_values(self):
        result = EvalResult({})
        self.assertFalse(result.success)
        self.assertEqual(result.score_us, -1_000_000.0)
        self.assertIsNone(result.error)
        self.assertIsNone(result.get_traceback())
        self.assertEqual(result.get_queue_time(), 0.0)
        self.assertEqual(result.get_eval_time(), 0.0)
        self.assertEqual(result.get_failed_tests(), [])

    def test_repr_success(self):
        result = EvalResult({"success": True, "score_us": 42.0})
        self.assertIn("success=True", repr(result))
        self.assertIn("42.0", repr(result))

    def test_repr_failure(self):
        result = EvalResult({"success": False, "error_type": "timeout"})
        self.assertIn("success=False", repr(result))
        self.assertIn("timeout", repr(result))


class MockServerHandler(BaseHTTPRequestHandler):
    """Mock HTTP server for client tests."""

    response_code = 200
    response_body = {}
    call_count = 0
    lock = threading.Lock()

    def do_POST(self):
        with self.lock:
            MockServerHandler.call_count += 1
            count = MockServerHandler.call_count

        if self.path == "/eval":
            if count <= getattr(MockServerHandler, "fail_first_n", 0):
                self.send_response(503)
                body = json.dumps({"error": "queue full"}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = json.dumps(MockServerHandler.response_body).encode()
            self.send_response(MockServerHandler.response_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class TestEvalClientBasic(unittest.TestCase):
    """Test client eval and health methods."""

    @classmethod
    def setUpClass(cls):
        MockServerHandler.response_code = 200
        MockServerHandler.response_body = {
            "success": True,
            "score_us": 42.0,
            "error": None,
            "error_type": None,
        }
        MockServerHandler.call_count = 0
        MockServerHandler.fail_first_n = 0

        cls.server = HTTPServer(("127.0.0.1", 18100), MockServerHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        MockServerHandler.call_count = 0
        MockServerHandler.fail_first_n = 0
        MockServerHandler.response_body = {
            "success": True,
            "score_us": 42.0,
            "error": None,
            "error_type": None,
        }

    def test_eval_success(self):
        client = EvalClient("http://127.0.0.1:18100")
        result = client.eval(code="print(1)", task_name="test")
        self.assertIsInstance(result, EvalResult)
        self.assertTrue(result.success)
        self.assertEqual(result.score_us, 42.0)

    def test_health(self):
        client = EvalClient("http://127.0.0.1:18100")
        health = client.health()
        self.assertEqual(health["status"], "ok")

    def test_eval_batch(self):
        client = EvalClient("http://127.0.0.1:18100")
        items = [
            {"code": "a", "task_name": "t1"},
            {"code": "b", "task_name": "t2"},
        ]
        results = client.eval_batch(items)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, EvalResult) for r in results))

    def test_base_url_trailing_slash(self):
        client = EvalClient("http://127.0.0.1:18100/")
        self.assertEqual(client.base_url, "http://127.0.0.1:18100")


class TestEvalClientRetry(unittest.TestCase):
    """Test client retry logic on 503."""

    @classmethod
    def setUpClass(cls):
        MockServerHandler.response_code = 200
        MockServerHandler.response_body = {
            "success": True,
            "score_us": 99.0,
        }
        MockServerHandler.call_count = 0
        MockServerHandler.fail_first_n = 0

        cls.server = HTTPServer(("127.0.0.1", 18101), MockServerHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        MockServerHandler.call_count = 0
        MockServerHandler.fail_first_n = 0

    def test_retry_on_503_then_success(self):
        MockServerHandler.fail_first_n = 1
        client = EvalClient("http://127.0.0.1:18101", max_retries=3)
        result = client.eval(code="x", task_name="t")
        self.assertTrue(result.success)
        self.assertGreaterEqual(MockServerHandler.call_count, 2)

    def test_connection_error_raises(self):
        client = EvalClient("http://127.0.0.1:19999", max_retries=1)
        with self.assertRaises(ConnectionError):
            client.eval(code="x", task_name="t")


if __name__ == "__main__":
    unittest.main()
