"""Tests for the shared eval pool — queue bounding, requeue, timing."""

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from eval_server.container import ContainerHealth, PooledKernelEvaluator
from eval_server.pool import EvalRequest, SharedEvalPool


class FakeEvaluator:
    """Minimal stand-in for PooledKernelEvaluator for unit tests."""

    def __init__(self, gpu_id, result=None, delay=0):
        self.gpu_id = gpu_id
        self._result = result
        self._delay = delay
        self.health = ContainerHealth.HEALTHY
        self.eval_count = 0
        self.restart_count = 0

    def start(self):
        pass

    def stop(self):
        pass

    def evaluate(self, code, task_name, same_container_retry=0):
        if self._delay:
            time.sleep(self._delay)
        self.eval_count += 1
        return self._result, None

    def get_status(self):
        return {
            "gpu_id": self.gpu_id,
            "health": self.health.value,
            "eval_count": self.eval_count,
            "restart_count": self.restart_count,
        }


class TestQueueBounding(unittest.TestCase):
    """Test that the queue rejects when full with 503."""

    def test_queue_full_rejects(self):
        evaluator = FakeEvaluator(0, delay=0.5, result={
            "success": True,
            "score_us": 100.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
            "benchmark_details": None,
        })

        pool = SharedEvalPool([evaluator], {0: "Tesla T4"}, eval_timeout=10)
        pool.start()

        try:
            accepted = 0
            rejected = 0
            for i in range(20):
                req = EvalRequest(f"code_{i}", "task")
                if pool.submit(req):
                    accepted += 1
                else:
                    rejected += 1

            self.assertGreater(rejected, 0, "Some requests should be rejected when queue is full")
            self.assertEqual(pool._max_queue_depth, 8)  # 1 GPU * 8
        finally:
            pool.stop()

    def test_queue_maxsize_scales_with_gpus(self):
        evaluators = [FakeEvaluator(i) for i in range(4)]
        pool = SharedEvalPool(evaluators, {i: "GPU" for i in range(4)})
        self.assertEqual(pool._max_queue_depth, 32)  # 4 GPUs * 8


class TestRequeueOnInfraFailure(unittest.TestCase):
    """Test that None result triggers requeue."""

    def test_requeue_on_none(self):
        call_count = [0]

        class FailOnceEvaluator(FakeEvaluator):
            def evaluate(self, code, task_name, same_container_retry=0):
                call_count[0] += 1
                if call_count[0] == 1:
                    return None, None
                return {
                    "success": True,
                    "score_us": 42.0,
                    "error": None,
                    "error_type": None,
                    "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
                    "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
                    "benchmark_details": None,
                }, None

        evaluator = FailOnceEvaluator(0)
        pool = SharedEvalPool([evaluator], {0: "GPU"}, eval_timeout=10)
        pool.start()

        try:
            req = EvalRequest("code", "task")
            pool.submit(req)
            req.done.wait(timeout=5)

            self.assertIsNotNone(req.result)
            self.assertEqual(req.retry_count, 1)
            self.assertTrue(req.result["success"])
        finally:
            pool.stop()

    def test_infra_failure_after_max_retries(self):
        evaluator = FakeEvaluator(0, result=None)
        pool = SharedEvalPool([evaluator], {0: "GPU"}, eval_timeout=10)
        pool.start()

        try:
            req = EvalRequest("code", "task")
            pool.submit(req)
            req.done.wait(timeout=10)

            self.assertIsNotNone(req.result)
            self.assertFalse(req.result["success"])
            self.assertEqual(req.result["error_type"], "infra_failure")
        finally:
            pool.stop()


class TestTimingCapture(unittest.TestCase):
    """Test that all 5 timestamps are populated."""

    def test_all_timestamps_populated(self):
        evaluator = FakeEvaluator(0, result={
            "success": True,
            "score_us": 100.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
            "benchmark_details": None,
        })

        pool = SharedEvalPool([evaluator], {0: "GPU"}, eval_timeout=10)
        pool.start()

        try:
            req = EvalRequest("code", "task")
            pool.submit(req)
            req.done.wait(timeout=5)

            self.assertIsNotNone(req.result)
            timing = req.result.get("timing")
            self.assertIsNotNone(timing)

            timestamps = timing["timestamps"]
            for key in ["received", "gpu_assigned", "eval_started", "eval_completed", "response_sent"]:
                self.assertIsNotNone(timestamps.get(key), f"Missing timestamp: {key}")

            self.assertGreater(timing["total_time_ms"], 0)
            self.assertGreaterEqual(timing["queue_time_ms"], 0)
            self.assertGreaterEqual(timing["eval_time_ms"], 0)
        finally:
            pool.stop()


class TestGPUTypeValidation(unittest.TestCase):
    """Test GPU type matching and mismatch rejection."""

    def test_matching_gpu_type(self):
        pool = SharedEvalPool(
            [FakeEvaluator(0)],
            {0: "NVIDIA H100"},
        )
        matches = pool.get_matching_gpus("H100")
        self.assertEqual(matches, [0])

    def test_case_insensitive_match(self):
        pool = SharedEvalPool(
            [FakeEvaluator(0)],
            {0: "NVIDIA H100 SXM"},
        )
        matches = pool.get_matching_gpus("h100")
        self.assertEqual(matches, [0])

    def test_no_match(self):
        pool = SharedEvalPool(
            [FakeEvaluator(0)],
            {0: "NVIDIA A100"},
        )
        matches = pool.get_matching_gpus("H100")
        self.assertEqual(matches, [])

    def test_none_gpu_type_matches_all(self):
        pool = SharedEvalPool(
            [FakeEvaluator(0), FakeEvaluator(1)],
            {0: "NVIDIA A100", 1: "NVIDIA H100"},
        )
        matches = pool.get_matching_gpus(None)
        self.assertEqual(matches, [0, 1])

    def test_multiple_matching_gpus(self):
        pool = SharedEvalPool(
            [FakeEvaluator(0), FakeEvaluator(1), FakeEvaluator(2)],
            {0: "NVIDIA H100", 1: "NVIDIA H100", 2: "NVIDIA A100"},
        )
        matches = pool.get_matching_gpus("H100")
        self.assertEqual(matches, [0, 1])


class TestEvalRequestTiming(unittest.TestCase):
    """Test EvalRequest timing methods."""

    def test_initial_state(self):
        req = EvalRequest("code", "task")
        self.assertIsNotNone(req.ts_received)
        self.assertIsNone(req.ts_gpu_assigned)
        self.assertEqual(req.queue_time_ms(), 0.0)
        self.assertEqual(req.eval_time_ms(), 0.0)

    def test_timing_progresses(self):
        req = EvalRequest("code", "task")
        time.sleep(0.01)
        req.set_gpu_assigned()
        self.assertGreater(req.queue_time_ms(), 0)

        time.sleep(0.01)
        req.set_eval_started()
        req.set_eval_completed()
        self.assertGreater(req.eval_time_ms(), 0)

        req.set_response_sent()
        self.assertGreater(req.total_time_ms(), 0)

    def test_timing_dict_structure(self):
        req = EvalRequest("code", "task")
        req.set_gpu_assigned()
        req.set_eval_started()
        req.set_eval_completed()
        req.set_response_sent()

        d = req.timing_dict()
        self.assertIn("queue_time_ms", d)
        self.assertIn("eval_time_ms", d)
        self.assertIn("total_time_ms", d)
        self.assertIn("timestamps", d)
        self.assertEqual(len(d["timestamps"]), 5)


if __name__ == "__main__":
    unittest.main()
