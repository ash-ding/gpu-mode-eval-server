"""Tests for the container pool worker logic (without GPU/container)."""

import json
import os
import sys
import tempfile
import unittest


WORKER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "container")
sys.path.insert(0, WORKER_DIR)
import pool_worker


class TestMakeErrorResult(unittest.TestCase):
    """Test error result factory."""

    def test_basic_error(self):
        result = pool_worker.make_error_result("something broke", "eval_failure")
        self.assertFalse(result["success"])
        self.assertEqual(result["score_us"], -1_000_000.0)
        self.assertEqual(result["error"], "something broke")
        self.assertEqual(result["error_type"], "eval_failure")
        self.assertIsNotNone(result["logs"])
        self.assertIsNotNone(result["test_results"])
        self.assertIsNone(result["benchmark_details"])

    def test_with_custom_logs(self):
        logs = {"stdout": "out", "stderr": "err", "compilation_log": "", "traceback": "tb"}
        result = pool_worker.make_error_result("fail", "compilation_error", logs=logs)
        self.assertEqual(result["logs"]["stdout"], "out")
        self.assertEqual(result["logs"]["traceback"], "tb")


class TestCompileKernel(unittest.TestCase):
    """Test kernel compilation."""

    def test_valid_code(self):
        ns, logs = pool_worker.compile_kernel("x = 42", {})
        self.assertIsNotNone(ns)
        self.assertEqual(ns["x"], 42)
        self.assertIsNone(logs["traceback"])

    def test_syntax_error(self):
        ns, logs = pool_worker.compile_kernel("def f(:", {})
        self.assertIsNone(ns)
        self.assertIsNotNone(logs["traceback"])
        self.assertIn("SyntaxError", logs["traceback"])

    def test_runtime_error(self):
        ns, logs = pool_worker.compile_kernel("raise ValueError('boom')", {})
        self.assertIsNone(ns)
        self.assertIn("ValueError", logs["traceback"])

    def test_captures_stdout(self):
        ns, logs = pool_worker.compile_kernel("print('hello')", {})
        self.assertIsNotNone(ns)
        self.assertIn("hello", logs["stdout"])


class TestEvaluate(unittest.TestCase):
    """Test the evaluate function."""

    def test_missing_code(self):
        result = pool_worker.evaluate({"task_name": "test"})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No code provided")

    def test_missing_task_name(self):
        result = pool_worker.evaluate({"code": "x = 1"})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No task_name provided")

    def test_nonexistent_task(self):
        result = pool_worker.evaluate({"code": "x = 1", "task_name": "nonexistent_xyz"})
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])


class TestResponseFormat(unittest.TestCase):
    """Test that all required fields are in the response."""

    def test_error_response_has_all_fields(self):
        result = pool_worker.make_error_result("err", "eval_failure")

        self.assertIn("success", result)
        self.assertIn("score_us", result)
        self.assertIn("error", result)
        self.assertIn("error_type", result)
        self.assertIn("logs", result)
        self.assertIn("test_results", result)
        self.assertIn("benchmark_details", result)

        logs = result["logs"]
        self.assertIn("stdout", logs)
        self.assertIn("stderr", logs)
        self.assertIn("compilation_log", logs)
        self.assertIn("traceback", logs)

        tr = result["test_results"]
        self.assertIn("passed", tr)
        self.assertIn("failed", tr)
        self.assertIn("total", tr)
        self.assertIn("first_failure", tr)
        self.assertIn("details", tr)

    def test_compilation_error_response(self):
        result = pool_worker.evaluate({"code": "def f(:", "task_name": "nonexistent_xyz"})
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "compilation_error")
        self.assertIsNotNone(result["logs"]["traceback"])


if __name__ == "__main__":
    unittest.main()
