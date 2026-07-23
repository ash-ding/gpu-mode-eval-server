"""Shared eval pool — bounded queue with per-GPU worker threads."""

import hashlib
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .container import PooledKernelEvaluator
from .metrics import metrics

logger = logging.getLogger(__name__)

MAX_INFRA_RETRIES = 3

# Request counter for generating unique IDs
_request_counter = 0
_request_counter_lock = threading.Lock()


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    global _request_counter
    with _request_counter_lock:
        _request_counter += 1
        return f"req_{_request_counter:06d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_hash(code: str) -> str:
    """Generate a short hash of the code for logging."""
    return hashlib.sha256(code.encode()).hexdigest()[:8]


class EvalRequest:
    """A single evaluation request with timing metadata."""

    def __init__(self, code: str, task_name: str, gpu_type: Optional[str] = None):
        self.request_id = _generate_request_id()
        self.code = code
        self.code_hash = _code_hash(code)
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

            # Log request received by worker
            logger.info(
                "Request received: request_id=%s task=%s gpu_type=%s code_hash=%s queue_depth=%d",
                request.request_id,
                request.task_name,
                request.gpu_type or "any",
                request.code_hash,
                self._queue.qsize(),
            )

            request.set_gpu_assigned()

            if request.gpu_type is not None:
                matching = self.get_matching_gpus(request.gpu_type)
                if evaluator.gpu_id not in matching:
                    try:
                        self._queue.put_nowait(request)
                    except queue.Full:
                        logger.warning(
                            'Requeue failed due to full queue for GPU type %s. '
                            'Pathological state: all GPUs wrong type AND queue saturated. '
                            'request_id=%s',
                            request.gpu_type,
                            request.request_id,
                        )
                        request.result = self._gpu_mismatch_result(request, evaluator)
                        request.set_response_sent()
                        request.done.set()
                    continue

            # Log request assigned to GPU
            logger.info(
                "Request assigned: request_id=%s task=%s gpu_id=%d queue_time_ms=%.1f",
                request.request_id,
                request.task_name,
                evaluator.gpu_id,
                request.queue_time_ms(),
            )

            # Queue depth warning
            current_depth = self._queue.qsize()
            if current_depth > self._max_queue_depth * 0.8:
                logger.warning(
                    "Queue filling up: %d/%d (%.1f%%) - consider adding more GPUs",
                    current_depth,
                    self._max_queue_depth,
                    100.0 * current_depth / self._max_queue_depth,
                )

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
                        "Infra failure on GPU %d, requeuing to different GPU (attempt %d/%d) request_id=%s task=%s",
                        evaluator.gpu_id, request.retry_count, MAX_INFRA_RETRIES,
                        request.request_id, request.task_name,
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
                    logger.error(
                        "Infra failure exhausted retries: request_id=%s task=%s gpu_id=%d retries=%d",
                        request.request_id, request.task_name, evaluator.gpu_id, request.retry_count,
                    )

            request.set_response_sent()

            metrics.observe_latency(
                task=request.task_name,
                gpu=evaluator.gpu_id,
                latency_seconds=request.eval_time_ms() / 1000.0,
            )
            metrics.inc_evals(
                success=result.get("success", False),
                error_type=result.get("error_type"),
            )
            metrics.set_queue_depth(self._queue.qsize())
            metrics.set_gpu_state(evaluator.gpu_id, evaluator.health.value)

            result["timing"] = request.timing_dict()
            result["metadata"] = {
                "gpu_id": evaluator.gpu_id,
                "gpu_name": self.gpu_names.get(evaluator.gpu_id, "unknown"),
                "container_restarts": evaluator.restart_count,
                "retry_count": request.retry_count,
                "same_container_retry": request.same_container_retry,
                "different_gpu_retry": request.different_gpu_retry,
                "crash_signal": request.crash_signal,
                "request_id": request.request_id,
            }

            # Log eval completion with detailed information
            success = result.get("success", False)
            error_type = result.get("error_type")
            test_results = result.get("test_results", {})

            if success:
                # Success case - log with INFO level
                logger.info(
                    "Eval completed: request_id=%s task=%s gpu_id=%d success=True "
                    "score_us=%.1f duration_ms=%.1f tests_passed=%d/%d queue_time_ms=%.1f",
                    request.request_id,
                    request.task_name,
                    evaluator.gpu_id,
                    result.get("score_us", 0),
                    request.eval_time_ms(),
                    test_results.get("passed", 0),
                    test_results.get("total", 0),
                    request.queue_time_ms(),
                )
            else:
                # Failure case - log with appropriate level based on error type
                if error_type == "compilation_error":
                    error_msg = result.get("error", "Unknown error")[:200]
                    logger.error(
                        "Compilation failed: request_id=%s task=%s gpu_id=%d error_type=%s error=%s",
                        request.request_id,
                        request.task_name,
                        evaluator.gpu_id,
                        error_type,
                        error_msg,
                    )
                elif error_type == "eval_failure":
                    # Test failures or runtime errors
                    logger.warning(
                        "Eval failed: request_id=%s task=%s gpu_id=%d error_type=%s "
                        "tests_passed=%d/%d error=%s",
                        request.request_id,
                        request.task_name,
                        evaluator.gpu_id,
                        error_type,
                        test_results.get("passed", 0),
                        test_results.get("total", 0),
                        result.get("error", "Unknown")[:100],
                    )
                elif error_type == "timeout":
                    logger.error(
                        "Eval timeout: request_id=%s task=%s gpu_id=%d duration_ms=%.1f",
                        request.request_id,
                        request.task_name,
                        evaluator.gpu_id,
                        request.eval_time_ms(),
                    )
                else:
                    # Other error types (infra_failure, gpu_mismatch, queue_full, etc.)
                    logger.error(
                        "Eval failed: request_id=%s task=%s gpu_id=%d error_type=%s error=%s",
                        request.request_id,
                        request.task_name,
                        evaluator.gpu_id,
                        error_type or "unknown",
                        result.get("error", "Unknown")[:200],
                    )

            # Slow eval warning (only for successful evals)
            if success and request.eval_time_ms() > 60000:  # > 1 minute
                logger.warning(
                    "Slow eval detected: request_id=%s task=%s gpu_id=%d duration_ms=%.1f (>60s)",
                    request.request_id,
                    request.task_name,
                    evaluator.gpu_id,
                    request.eval_time_ms(),
                )

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
