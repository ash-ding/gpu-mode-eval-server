"""Container manager — spawns and manages persistent GPU containers."""

import json
import logging
import subprocess
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

RESTART_INTERVAL = 1000


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
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._container_name = f"eval-gpu-{gpu_id}"

    def start(self):
        """Start the container process."""
        self._kill_existing()

        cmd = [
            self.runtime.runtime,
            "run",
            "--rm",
            "-i",
            "--name", self._container_name,
        ]
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
        self.health = ContainerHealth.HEALTHY
        logger.info("Started container for GPU %d (pid=%d)", self.gpu_id, self._proc.pid)

    def _kill_existing(self):
        """Kill any existing container with our name."""
        try:
            subprocess.run(
                [self.runtime.runtime, "rm", "-f", self._container_name],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None

    def evaluate(self, code: str, task_name: str) -> Optional[dict]:
        """Send an eval request and return the result, or None on infra failure."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._try_restart()
                if self._proc is None or self._proc.poll() is not None:
                    self.health = ContainerHealth.FAILED
                    return None

            if self.eval_count > 0 and self.eval_count % RESTART_INTERVAL == 0:
                logger.info(
                    "Periodic restart for GPU %d after %d evals",
                    self.gpu_id, self.eval_count,
                )
                self._try_restart()
                if self._proc is None or self._proc.poll() is not None:
                    self.health = ContainerHealth.FAILED
                    return None

            request = json.dumps({"code": code, "task_name": task_name}) + "\n"

            try:
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.error("Container stdin broken for GPU %d", self.gpu_id)
                self.health = ContainerHealth.RECOVERING
                return None

            try:
                response_line = self._read_with_timeout(self.timeout)
            except TimeoutError:
                logger.error("Container timeout for GPU %d", self.gpu_id)
                self.health = ContainerHealth.RECOVERING
                self._kill_existing()
                return None

            if response_line is None:
                logger.error("Container stdout closed for GPU %d", self.gpu_id)
                self.health = ContainerHealth.RECOVERING
                return None

            if self._proc.poll() is not None and self._proc.returncode != 0:
                logger.error(
                    "Container exited with code %d for GPU %d",
                    self._proc.returncode, self.gpu_id,
                )
                self.health = ContainerHealth.RECOVERING
                return None

            try:
                result = json.loads(response_line)
            except json.JSONDecodeError:
                logger.error("Invalid JSON from container GPU %d: %s", self.gpu_id, response_line[:200])
                self.health = ContainerHealth.RECOVERING
                return None

            self.eval_count += 1
            self.health = ContainerHealth.HEALTHY
            return result

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

    def _try_restart(self):
        """Attempt to restart the container."""
        logger.info("Restarting container for GPU %d", self.gpu_id)
        self.health = ContainerHealth.RECOVERING
        try:
            self._kill_existing()
            self.start()
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
