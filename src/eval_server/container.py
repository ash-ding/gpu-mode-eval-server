"""Container manager — spawns and manages persistent GPU containers."""

import json
import logging
import os
import subprocess
import threading
import time
from enum import Enum
from typing import Optional

from .failure_detection import check_gpu_health

logger = logging.getLogger(__name__)

RESTART_INTERVAL = 1000

KERNEL_BUG_SIGNALS = {
    -9: "SIGKILL/OOM",
    -11: "SIGSEGV",
    -6: "SIGABRT",
}


def check_crash_signature(returncode: int) -> tuple[str, bool]:
    """Classify a crash by its signal.

    Returns (description, is_kernel_bug).
    Kernel bugs (OOM, segfault, abort) are definitive eval failures —
    retrying won't help.
    """
    if returncode in KERNEL_BUG_SIGNALS:
        return KERNEL_BUG_SIGNALS[returncode], True
    return f"exit code {returncode}", False


class ContainerHealth(Enum):
    HEALTHY = "healthy"
    RECOVERING = "recovering"
    FAILED = "failed"


class ContainerRuntime:
    """Auto-detects podman or docker and abstracts GPU flag differences."""

    def __init__(self):
        self.runtime = self._detect()

    def _detect(self):
        for cmd in ("podman", "docker"):
            try:
                subprocess.run(
                    [cmd, "version"],
                    capture_output=True,
                    timeout=10,
                )
                logger.info("Detected container runtime: %s", cmd)
                return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        raise RuntimeError("No container runtime found. Install podman or docker.")

    def gpu_flag(self, gpu_id: int) -> list[str]:
        if self.runtime == "podman":
            return ["--device", f"nvidia.com/gpu={gpu_id}"]
        return ["--gpus", f"device={gpu_id}"]


def detect_container_runtime() -> ContainerRuntime:
    return ContainerRuntime()


def query_gpu_names(gpu_ids: list[int]) -> dict[int, str]:
    """Query nvidia-smi for GPU model names. Returns {gpu_id: name}."""
    result = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                parts = line.split(", ", 1)
                if len(parts) == 2:
                    idx = int(parts[0].strip())
                    name = parts[1].strip()
                    if idx in gpu_ids:
                        result[idx] = name
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        logger.warning("nvidia-smi not available, GPU names unknown")
    return result


class PooledKernelEvaluator:
    """Manages a persistent container for a single GPU.

    Lifecycle: spawn container -> feed evals via stdin/stdout JSON ->
    periodic restart every RESTART_INTERVAL evals -> kill on shutdown.
    """

    def __init__(
        self,
        gpu_id: int,
        runtime: ContainerRuntime,
        image: str,
        timeout: int = 530,
        tasks_dir: Optional[str] = None,
    ):
        self.gpu_id = gpu_id
        self.runtime = runtime
        self.image = image
        self.timeout = timeout
        self.tasks_dir = tasks_dir

        self.health = ContainerHealth.HEALTHY
        self.eval_count = 0
        self.restart_count = 0
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._container_name = f"eval-gpu-{gpu_id}"

    def start(self, _skip_cleanup=False):
        """Start the container process."""
        if not _skip_cleanup:
            self._kill_existing()

        cmd = [
            self.runtime.runtime,
            "run",
            "-i",
            "--name", self._container_name,
        ]
        if self.runtime.runtime == "podman":
            cmd.append("--replace")
        cmd.extend(self.runtime.gpu_flag(self.gpu_id))

        if self.tasks_dir:
            cmd.extend(["-v", f"{self.tasks_dir}:/workspace/lib:ro"])

        cmd.append(self.image)

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.5)
        if self._proc.poll() is not None:
            stderr = self._read_stderr()
            logger.error(
                "Container failed to start for GPU %d (exit=%d): %s",
                self.gpu_id, self._proc.returncode, stderr[:500],
            )
            self.health = ContainerHealth.FAILED
            self._proc = None
            return

        self.health = ContainerHealth.HEALTHY
        logger.info("Started container for GPU %d (pid=%d)", self.gpu_id, self._proc.pid)

    def _kill_existing(self):
        """Kill any existing container with our name."""
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None

        for attempt in range(3):
            try:
                subprocess.run(
                    [self.runtime.runtime, "rm", "-f", self._container_name],
                    capture_output=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            try:
                check = subprocess.run(
                    [self.runtime.runtime, "container", "exists", self._container_name],
                    capture_output=True,
                    timeout=5,
                )
                if check.returncode != 0:
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                break
            if attempt < 2:
                time.sleep(1)

        self._kill_gpu_processes()

    def _kill_gpu_processes(self):
        """Kill orphan host processes still holding this GPU's memory."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 f"--id={self.gpu_id}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return
            for line in result.stdout.strip().splitlines():
                pid_str = line.strip()
                if not pid_str:
                    continue
                try:
                    pid = int(pid_str)
                    os.kill(pid, 9)
                    logger.info("Killed orphan GPU process %d on GPU %d", pid, self.gpu_id)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("nvidia-smi not available for GPU cleanup on GPU %d", self.gpu_id)

    def evaluate(self, code: str, task_name: str, same_container_retry: int = 0) -> tuple[Optional[dict], Optional[int]]:
        """Send an eval request and return (result, crash_signal).

        Returns:
            (result_dict, None) on success or definitive eval failure.
            (None, returncode) on infrastructure failure — the caller
            decides whether to retry.
        """
        with self._lock:
            healthy, health_err = check_gpu_health(self.gpu_id)
            if not healthy:
                logger.warning("GPU %d unhealthy: %s", self.gpu_id, health_err)
                self.health = ContainerHealth.RECOVERING
                return None, None

            if self._proc is None or self._proc.poll() is not None:
                self._try_restart()
                if self._proc is None or self._proc.poll() is not None:
                    self.health = ContainerHealth.FAILED
                    return None, None

            if self.eval_count > 0 and self.eval_count % RESTART_INTERVAL == 0:
                logger.info(
                    "Periodic restart for GPU %d after %d evals",
                    self.gpu_id, self.eval_count,
                )
                self._try_restart()
                if self._proc is None or self._proc.poll() is not None:
                    self.health = ContainerHealth.FAILED
                    return None, None

            request = json.dumps({"code": code, "task_name": task_name}) + "\n"

            try:
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                stderr = self._read_stderr()
                logger.error("Container stdin broken for GPU %d: %s", self.gpu_id, stderr[:500] if stderr else "no stderr")
                self.health = ContainerHealth.RECOVERING
                return None, None

            try:
                response_line = self._read_with_timeout(self.timeout)
            except TimeoutError:
                self.health = ContainerHealth.RECOVERING
                self._kill_existing()
                logger.error(
                    "Container timeout for GPU %d — kernel hung (not infra failure, no retry)",
                    self.gpu_id,
                )
                self.eval_count += 1
                return {
                    "success": False,
                    "score_us": -1_000_000.0,
                    "error": f"Kernel timed out after {self.timeout}s on GPU {self.gpu_id}",
                    "error_type": "timeout",
                    "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
                    "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
                    "benchmark_details": None,
                }, None

            if response_line is None:
                stderr = self._read_stderr()
                logger.error("Container stdout closed for GPU %d: %s", self.gpu_id, stderr[:500] if stderr else "no stderr")
                self.health = ContainerHealth.RECOVERING
                returncode = self._proc.returncode if self._proc.poll() is not None else None
                return None, returncode

            if self._proc.poll() is not None and self._proc.returncode != 0:
                returncode = self._proc.returncode
                desc, is_kernel_bug = check_crash_signature(returncode)
                logger.error(
                    "Container exited with %s (code %d) for GPU %d, task=%s, is_kernel_bug=%s",
                    desc, returncode, self.gpu_id, task_name, is_kernel_bug,
                )
                self.health = ContainerHealth.RECOVERING

                if is_kernel_bug:
                    logger.warning(
                        "Kernel bug detected: task=%s gpu_id=%d signal=%s - marking as eval_failure (no retry)",
                        task_name, self.gpu_id, desc,
                    )
                    return {
                        "success": False,
                        "score_us": -1_000_000.0,
                        "error": f"Kernel crashed with {desc}",
                        "error_type": "eval_failure",
                        "logs": {"stdout": "", "stderr": response_line, "compilation_log": "", "traceback": None},
                        "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
                        "benchmark_details": None,
                    }, returncode

                if same_container_retry == 0:
                    logger.info(
                        "Container crashed (code %d), restarting and retrying on same GPU %d, task=%s",
                        returncode, self.gpu_id, task_name,
                    )
                    self._try_restart()
                    return self.evaluate(code, task_name, same_container_retry=1)

                return None, returncode

            try:
                result = json.loads(response_line)
            except json.JSONDecodeError:
                logger.error("Invalid JSON from container GPU %d: %s", self.gpu_id, response_line[:200])
                self.health = ContainerHealth.RECOVERING
                return None, None

            self.eval_count += 1
            self.health = ContainerHealth.HEALTHY
            return result, None

    def _read_with_timeout(self, timeout: int) -> Optional[str]:
        """Read a line from container stdout with timeout."""
        result = [None]
        error = [None]

        def _read():
            try:
                line = self._proc.stdout.readline()
                result[0] = line if line else None
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(f"Container read timeout after {timeout}s")

        if error[0] is not None:
            return None

        return result[0].rstrip("\n") if result[0] else None

    def _read_stderr(self, timeout: float = 2.0) -> str:
        """Read available stderr without blocking (thread with timeout)."""
        result = [""]
        def _read():
            try:
                if self._proc and self._proc.stderr:
                    result[0] = self._proc.stderr.read()[:1000]
            except Exception:
                pass
        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        return result[0]

    def _try_restart(self):
        """Attempt to restart the container."""
        logger.info("Restarting container for GPU %d", self.gpu_id)
        self.health = ContainerHealth.RECOVERING
        try:
            self._kill_existing()
            self.start(_skip_cleanup=True)
            self.restart_count += 1
        except Exception:
            logger.exception("Failed to restart container for GPU %d", self.gpu_id)
            self.health = ContainerHealth.FAILED

    def stop(self):
        """Stop the container."""
        self._kill_existing()
        self.health = ContainerHealth.FAILED

    def get_status(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "health": self.health.value,
            "eval_count": self.eval_count,
            "restart_count": self.restart_count,
        }
