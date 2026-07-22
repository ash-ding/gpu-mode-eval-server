"""Tests for container manager — returncode validation, health states."""

import json
import subprocess
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from eval_server.container import (
    ContainerHealth,
    ContainerRuntime,
    PooledKernelEvaluator,
    query_gpu_names,
)


class TestContainerRuntimeDetection(unittest.TestCase):
    """Test container runtime auto-detection."""

    @patch("subprocess.run")
    def test_detects_podman_first(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        runtime = ContainerRuntime()
        self.assertEqual(runtime.runtime, "podman")

    @patch("subprocess.run")
    def test_podman_gpu_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        runtime = ContainerRuntime()
        runtime.runtime = "podman"
        self.assertEqual(runtime.gpu_flag(3), ["--device", "nvidia.com/gpu=3"])

    def test_docker_gpu_flag(self):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "docker"
        self.assertEqual(runtime.gpu_flag(2), ["--gpus", "device=2"])


class TestQueryGPUNames(unittest.TestCase):
    """Test nvidia-smi GPU name query."""

    @patch("subprocess.run")
    def test_parses_gpu_names(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA H100 SXM\n1, NVIDIA H100 SXM\n2, NVIDIA A100-SXM4-80GB\n",
        )
        names = query_gpu_names([0, 1, 2])
        self.assertEqual(names, {
            0: "NVIDIA H100 SXM",
            1: "NVIDIA H100 SXM",
            2: "NVIDIA A100-SXM4-80GB",
        })

    @patch("subprocess.run")
    def test_filters_to_requested_ids(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA H100\n1, NVIDIA A100\n",
        )
        names = query_gpu_names([0])
        self.assertEqual(names, {0: "NVIDIA H100"})

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_handles_missing_nvidia_smi(self, mock_run):
        names = query_gpu_names([0])
        self.assertEqual(names, {})


class TestContainerReturncodeValidation(unittest.TestCase):
    """Test that container returncode is validated before trusting results."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_nonzero_returncode_returns_none(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.health = ContainerHealth.HEALTHY

        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 1]  # alive during write, dead after read
        mock_proc.returncode = 1
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        mock_proc.stdout.readline.return_value = '{"success": true}\n'
        evaluator._proc = mock_proc

        with patch.object(evaluator, "_read_with_timeout", return_value='{"success": true}'):
            result = evaluator.evaluate("code", "task")

        self.assertIsNone(result)
        self.assertEqual(evaluator.health, ContainerHealth.RECOVERING)


class TestPeriodicRestart(unittest.TestCase):
    """Test periodic container restart every 1000 evals."""

    @patch.object(PooledKernelEvaluator, "_kill_existing")
    @patch.object(PooledKernelEvaluator, "start")
    def test_restart_at_interval(self, mock_start, mock_kill):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "podman"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")

        evaluator.eval_count = 1000
        evaluator.health = ContainerHealth.HEALTHY

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.returncode = 0
        mock_proc.stdin.write.return_value = None
        mock_proc.stdin.flush.return_value = None
        evaluator._proc = mock_proc

        with patch.object(evaluator, "_try_restart") as mock_restart, \
             patch.object(evaluator, "_read_with_timeout", return_value='{"success": true}'):
            mock_restart.side_effect = lambda: setattr(evaluator, '_proc', mock_proc)
            evaluator.evaluate("code", "task")
            mock_restart.assert_called_once()


class TestHealthStates(unittest.TestCase):
    """Test health state machine transitions."""

    def test_initial_health(self):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "docker"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        self.assertEqual(evaluator.health, ContainerHealth.HEALTHY)

    def test_get_status(self):
        runtime = ContainerRuntime.__new__(ContainerRuntime)
        runtime.runtime = "docker"
        evaluator = PooledKernelEvaluator(0, runtime, "test:latest")
        evaluator.eval_count = 42
        evaluator.restart_count = 3

        status = evaluator.get_status()
        self.assertEqual(status["gpu_id"], 0)
        self.assertEqual(status["health"], "healthy")
        self.assertEqual(status["eval_count"], 42)
        self.assertEqual(status["restart_count"], 3)


if __name__ == "__main__":
    unittest.main()
