"""Integration tests for Phase 2 reliability hardening."""

import time
import unittest
from unittest.mock import MagicMock, patch

from eval_server.container import (
    ContainerHealth,
    ContainerRuntime,
    PooledKernelEvaluator,
    check_crash_signature,
)
from eval_server.failure_detection import check_gpu_health, check_memory_leak
from eval_server.pool import EvalRequest, SharedEvalPool


class FakeEvaluator:
    """Minimal stand-in for PooledKernelEvaluator for pool tests."""

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


class TestCrashSignature(unittest.TestCase):
    """Test error signature classification."""

    def test_oom_sigkill(self):
        desc, is_kernel_bug = check_crash_signature(-9)
        self.assertTrue(is_kernel_bug)
        self.assertIn("SIGKILL", desc)

    def test_segfault(self):
        desc, is_kernel_bug = check_crash_signature(-11)
        self.assertTrue(is_kernel_bug)
        self.assertIn("SIGSEGV", desc)

    def test_sigabrt(self):
        desc, is_kernel_bug = check_crash_signature(-6)
        self.assertTrue(is_kernel_bug)
        self.assertIn("SIGABRT", desc)

    def test_other_signal_not_kernel_bug(self):
        desc, is_kernel_bug = check_crash_signature(-15)
        self.assertFalse(is_kernel_bug)
        self.assertIn("exit code", desc)

    def test_positive_exit_code_not_kernel_bug(self):
        desc, is_kernel_bug = check_crash_signature(1)
        self.assertFalse(is_kernel_bug)


class TestSameContainerRetry(unittest.TestCase):
    """Test that container crash triggers same-container retry."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_retry_succeeds_after_crash(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.health = ContainerHealth.HEALTHY

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 1]
        mock_proc.returncode = 1
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        evaluator._proc = mock_proc

        mock_proc2 = MagicMock()
        mock_proc2.poll.return_value = None
        mock_proc2.returncode = 0
        mock_proc2.stdin.write.return_value = None
        mock_proc2.stdin.flush.return_value = None

        call_count = [0]

        def fake_read(timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                return "crash output"
            return '{"success": true, "score_us": 42.0}'

        def restart_side_effect():
            evaluator._proc = mock_proc2
            evaluator.health = ContainerHealth.RECOVERING
            evaluator.restart_count += 1

        with patch.object(evaluator, "_read_with_timeout", side_effect=fake_read), \
             patch.object(evaluator, "_try_restart", side_effect=restart_side_effect), \
             patch("eval_server.container.check_gpu_health", return_value=(True, "")):
            result, crash_signal = evaluator.evaluate("code", "task")

        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertEqual(call_count[0], 2)


class TestErrorSignatureOOM(unittest.TestCase):
    """Test that OOM (returncode -9) is classified as eval_failure."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_oom_returns_eval_failure(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.health = ContainerHealth.HEALTHY

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, -9]
        mock_proc.returncode = -9
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        evaluator._proc = mock_proc

        with patch.object(evaluator, "_read_with_timeout", return_value="OOM killed"), \
             patch("eval_server.container.check_gpu_health", return_value=(True, "")):
            result, crash_signal = evaluator.evaluate("code", "task")

        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "eval_failure")
        self.assertIn("SIGKILL/OOM", result["error"])
        self.assertEqual(crash_signal, -9)


class TestErrorSignatureSegfault(unittest.TestCase):
    """Test that SEGFAULT (returncode -11) is classified as eval_failure."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_segfault_returns_eval_failure(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.health = ContainerHealth.HEALTHY

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, -11]
        mock_proc.returncode = -11
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        evaluator._proc = mock_proc

        with patch.object(evaluator, "_read_with_timeout", return_value="segfault"), \
             patch("eval_server.container.check_gpu_health", return_value=(True, "")):
            result, crash_signal = evaluator.evaluate("code", "task")

        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "eval_failure")
        self.assertIn("SIGSEGV", result["error"])
        self.assertEqual(crash_signal, -11)


class TestRetryBudgetExhaustion(unittest.TestCase):
    """Test that after max retries, request fails with final error."""

    def test_exhausts_retry_budget(self):
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
            self.assertEqual(req.retry_count, 3)
        finally:
            pool.stop()


class TestTimeoutCalculationParallel(unittest.TestCase):
    """Test that timeout calculation accounts for parallel GPUs."""

    def test_single_gpu_timeout(self):
        evaluator = FakeEvaluator(0)
        pool = SharedEvalPool([evaluator], {0: "GPU"}, eval_timeout=530)
        expected = (8 // 1 + 1) * 530 + 60
        self.assertEqual(pool._request_timeout, expected)

    def test_multi_gpu_timeout(self):
        evaluators = [FakeEvaluator(i) for i in range(4)]
        pool = SharedEvalPool(evaluators, {i: "GPU" for i in range(4)}, eval_timeout=530)
        expected = (32 // 4 + 1) * 530 + 60
        self.assertEqual(pool._request_timeout, expected)

    def test_timeout_much_less_than_serial(self):
        evaluators = [FakeEvaluator(i) for i in range(4)]
        pool = SharedEvalPool(evaluators, {i: "GPU" for i in range(4)}, eval_timeout=530)
        serial_timeout = 32 * 530 + 60
        self.assertLess(pool._request_timeout, serial_timeout)


class TestResponseMetadataFields(unittest.TestCase):
    """Test that response includes new retry metadata fields."""

    def test_metadata_has_retry_fields(self):
        result_data = {
            "success": True,
            "score_us": 100.0,
            "error": None,
            "error_type": None,
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 1, "failed": 0, "total": 1, "first_failure": None, "details": []},
            "benchmark_details": None,
        }
        evaluator = FakeEvaluator(0, result=result_data)
        pool = SharedEvalPool([evaluator], {0: "GPU"}, eval_timeout=10)
        pool.start()

        try:
            req = EvalRequest("code", "task")
            pool.submit(req)
            req.done.wait(timeout=5)

            self.assertIsNotNone(req.result)
            meta = req.result["metadata"]
            self.assertIn("same_container_retry", meta)
            self.assertIn("different_gpu_retry", meta)
            self.assertIn("crash_signal", meta)
            self.assertEqual(meta["same_container_retry"], 0)
            self.assertEqual(meta["different_gpu_retry"], 0)
            self.assertIsNone(meta["crash_signal"])
        finally:
            pool.stop()


class TestGPUHealthCheck(unittest.TestCase):
    """Test GPU health check function."""

    @patch("subprocess.run")
    def test_healthy_gpu(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, 45",
        )
        healthy, err = check_gpu_health(0)
        self.assertTrue(healthy)
        self.assertEqual(err, "")

    @patch("subprocess.run")
    def test_high_temperature(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, 90",
        )
        healthy, err = check_gpu_health(0)
        self.assertFalse(healthy)
        self.assertIn("temperature", err)

    @patch("subprocess.run")
    def test_ecc_errors(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5, 45",
        )
        healthy, err = check_gpu_health(0)
        self.assertFalse(healthy)
        self.assertIn("ECC", err)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_nvidia_smi(self, mock_run):
        healthy, err = check_gpu_health(0)
        self.assertTrue(healthy)

    @patch("subprocess.run")
    def test_nvidia_smi_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="GPU not found",
        )
        healthy, err = check_gpu_health(0)
        self.assertFalse(healthy)
        self.assertIn("nvidia-smi failed", err)


class TestMemoryLeakDetection(unittest.TestCase):
    """Test GPU memory leak detection."""

    @patch("subprocess.run")
    def test_no_leak(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1000",
        )
        has_leak, current = check_memory_leak(0, baseline_mb=500)
        self.assertFalse(has_leak)
        self.assertEqual(current, 1000)

    @patch("subprocess.run")
    def test_leak_detected(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5000",
        )
        has_leak, current = check_memory_leak(0, baseline_mb=500)
        self.assertTrue(has_leak)
        self.assertEqual(current, 5000)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_nvidia_smi(self, mock_run):
        has_leak, current = check_memory_leak(0, baseline_mb=500)
        self.assertFalse(has_leak)
        self.assertEqual(current, 0)


class TestGPUUnhealthyTriggersRequeue(unittest.TestCase):
    """Test that unhealthy GPU causes evaluate to return None for requeue."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_unhealthy_gpu_returns_none(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.health = ContainerHealth.HEALTHY
        evaluator._proc = MagicMock()
        evaluator._proc.poll.return_value = None

        with patch("eval_server.container.check_gpu_health", return_value=(False, "GPU overheating")):
            result, crash_signal = evaluator.evaluate("code", "task")

        self.assertIsNone(result)
        self.assertEqual(evaluator.health, ContainerHealth.RECOVERING)


class TestEvalRequestRetryFields(unittest.TestCase):
    """Test that EvalRequest has the new retry tracking fields."""

    def test_initial_retry_fields(self):
        req = EvalRequest("code", "task")
        self.assertEqual(req.same_container_retry, 0)
        self.assertEqual(req.different_gpu_retry, 0)
        self.assertIsNone(req.crash_signal)

    def test_retry_fields_in_pool_after_infra_failure(self):
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
            self.assertEqual(req.different_gpu_retry, 1)
            meta = req.result["metadata"]
            self.assertEqual(meta["different_gpu_retry"], 1)
        finally:
            pool.stop()


if __name__ == "__main__":
    unittest.main()
