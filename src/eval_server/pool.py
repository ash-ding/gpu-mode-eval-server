"""Shared eval pool — bounded queue with per-GPU worker threads."""

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .container import PooledKernelEvaluator

logger = logging.getLogger(__name__)

MAX_INFRA_RETRIES = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalRequest:
    """A single evaluation request with timing metadata."""

    def __init__(self, code: str, task_name: str, gpu_type: Optional[str] = None):
        self.code = code
        self.task_name = task_name
        self.gpu_type = gpu_type
        self.retry_count = 0
        self.same_container_retry = 0
        self.different_gpu_retry = 0
        self.crash_signal: Optional[int] = None
        self.result: Optional[dict] = None
        self.done = threading.Event()

        self.ts_received = _now_iso()
        self.ts_gpu_assigned: Optional[str] = None
        self.ts_eval_started: Optional[str] = None
        self.ts_eval_completed: Optional[str] = None
        self.ts_response_sent: Optional[str] = None

        self._t_received = time.monotonic()
        self._t_gpu_assigned: Optional[float] = None
        self._t_eval_completed: Optional[float] = None

    def set_gpu_assigned(self):
        self.ts_gpu_assigned = _now_iso()
        self._t_gpu_assigned = time.monotonic()

    def set_eval_started(self):
        self.ts_eval_started = _now_iso()

    def set_eval_completed(self):
        self.ts_eval_completed = _now_iso()
        self._t_eval_completed = time.monotonic()

    def set_response_sent(self):
        self.ts_response_sent = _now_iso()

    def queue_time_ms(self) -> float:
        if self._t_gpu_assigned is None:
            return 0.0
        return (self._t_gpu_assigned - self._t_received) * 1000

    def eval_time_ms(self) -> float:
        if self._t_gpu_assigned is None or self._t_eval_completed is None:
            return 0.0
        return (self._t_eval_completed - self._t_gpu_assigned) * 1000

    def total_time_ms(self) -> float:
        return (time.monotonic() - self._t_received) * 1000

    def timing_dict(self) -> dict:
        return {
            "queue_time_ms": round(self.queue_time_ms(), 2),
            "eval_time_ms": round(self.eval_time_ms(), 2),
            "total_time_ms": round(self.total_time_ms(), 2),
            "timestamps": {
                "received": self.ts_received,
                "gpu_assigned": self.ts_gpu_assigned,
                "eval_started": self.ts_eval_started,
                "eval_completed": self.ts_eval_completed,
                "response_sent": self.ts_response_sent,
            },
        }


class SharedEvalPool:
    """Bounded eval pool with one worker thread per GPU."""

    def __init__(
        self,
        evaluators: list[PooledKernelEvaluator],
        gpu_names: dict[int, str],
        eval_timeout: int = 530,
    ):
        self.evaluators = evaluators
        self.gpu_names = gpu_names
        self.eval_timeout = eval_timeout

        num_gpus = len(evaluators)
        max_depth = num_gpus * 8
        self._queue: queue.Queue[EvalRequest] = queue.Queue(maxsize=max_depth)
        self._max_queue_depth = max_depth

        self._request_timeout = (max_depth // num_gpus + 1) * eval_timeout + 60

        self._workers: list[threading.Thread] = []
        self._running = False
        self._start_time = time.monotonic()

    def start(self):
        """Start all evaluator containers and worker threads."""
        self._running = True
        for evaluator in self.evaluators:
            evaluator.start()

        for i, evaluator in enumerate(self.evaluators):
            t = threading.Thread(
                target=self._worker_loop,
                args=(evaluator,),
                name=f"gpu-worker-{evaluator.gpu_id}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        logger.info(
            "Eval pool started: %d GPUs, queue depth %d",
            len(self.evaluators), self._max_queue_depth,
        )

    def submit(self, request: EvalRequest) -> bool:
        """Submit an eval request. Returns False if queue is full."""
        try:
            self._queue.put_nowait(request)
            return True
        except queue.Full:
            return False

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def get_matching_gpus(self, gpu_type: Optional[str]) -> list[int]:
        """Return GPU IDs that match the requested type. Empty list = no match."""
        if gpu_type is None:
            return [e.gpu_id for e in self.evaluators]

        gpu_type_lower = gpu_type.lower()
        return [
            gpu_id
            for gpu_id, name in self.gpu_names.items()
            if gpu_type_lower in name.lower()
        ]

    def get_health(self) -> dict:
        return {
            "workers": [e.get_status() for e in self.evaluators],
            "queue_depth": self.queue_depth(),
            "max_queue_depth": self._max_queue_depth,
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "gpu_names": self.gpu_names,
        }

    def stop(self):
        """Drain queue and stop all workers."""
        self._running = False

        for _ in self.evaluators:
            sentinel = EvalRequest("", "")
            sentinel.code = ""
            try:
                self._queue.put(sentinel, timeout=1)
            except queue.Full:
                pass

        for t in self._workers:
            t.join(timeout=5)

        for evaluator in self.evaluators:
            evaluator.stop()

        logger.info("Eval pool stopped")

    def _worker_loop(self, evaluator: PooledKernelEvaluator):
        """Worker thread: pull from queue, evaluate, set result."""
        while self._running:
            try:
                request = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if not request.code and not self._running:
                break

            if not request.code:
                continue

            request.set_gpu_assigned()

            if request.gpu_type is not None:
                matching = self.get_matching_gpus(request.gpu_type)
                if evaluator.gpu_id not in matching:
                    try:
                        self._queue.put_nowait(request)
                    except queue.Full:
                        logger.warning(
                            'Requeue failed due to full queue for GPU type %s. '
                            'Pathological state: all GPUs wrong type AND queue saturated.',
                            request.gpu_type,
                        )
                        request.result = self._gpu_mismatch_result(request, evaluator)
                        request.set_response_sent()
                        request.done.set()
                    continue

            request.set_eval_started()
            result, crash_signal = evaluator.evaluate(
                request.code, request.task_name,
                same_container_retry=request.same_container_retry,
            )
            request.set_eval_completed()

            if crash_signal is not None:
                request.crash_signal = crash_signal
                request.same_container_retry = 1

            if result is None:
                request.retry_count += 1
                request.different_gpu_retry += 1
                if request.retry_count < MAX_INFRA_RETRIES:
                    logger.warning(
                        "Infra failure on GPU %d, requeuing to different GPU (attempt %d/%d)",
                        evaluator.gpu_id, request.retry_count, MAX_INFRA_RETRIES,
                    )
                    try:
                        self._queue.put_nowait(request)
                    except queue.Full:
                        request.result = self._infra_failure_result(request, evaluator)
                        request.set_response_sent()
                        request.done.set()
                    continue
                else:
                    result = self._infra_failure_result(request, evaluator)

            request.set_response_sent()

            result["timing"] = request.timing_dict()
            result["metadata"] = {
                "gpu_id": evaluator.gpu_id,
                "gpu_name": self.gpu_names.get(evaluator.gpu_id, "unknown"),
                "container_restarts": evaluator.restart_count,
                "retry_count": request.retry_count,
                "same_container_retry": request.same_container_retry,
                "different_gpu_retry": request.different_gpu_retry,
                "crash_signal": request.crash_signal,
            }

            request.result = result
            request.done.set()

    def _infra_failure_result(self, request: EvalRequest, evaluator: PooledKernelEvaluator) -> dict:
        return {
            "success": False,
            "score_us": -1_000_000.0,
            "error": f"Infrastructure failure after {request.retry_count} retries",
            "error_type": "infra_failure",
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
            "benchmark_details": None,
        }

    def _gpu_mismatch_result(self, request: EvalRequest, evaluator: PooledKernelEvaluator) -> dict:
        available = ", ".join(f"{k}: {v}" for k, v in self.gpu_names.items())
        return {
            "success": False,
            "score_us": -1_000_000.0,
            "error": f"Requested GPU type '{request.gpu_type}' not available. Available: {available}",
            "error_type": "gpu_mismatch",
            "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
            "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
            "benchmark_details": None,
        }
