"""Simple in-memory Prometheus-format metrics."""

import threading
from collections import defaultdict

LATENCY_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1200, float("inf"))


class Metrics:
    """Thread-safe in-memory metrics with Prometheus text rendering."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latency_buckets: dict[tuple, list[int]] = defaultdict(
            lambda: [0] * len(LATENCY_BUCKETS)
        )
        self._latency_sum: dict[tuple, float] = defaultdict(float)
        self._latency_count: dict[tuple, int] = defaultdict(int)
        self._evals_total: dict[tuple, int] = defaultdict(int)
        self._queue_depth = 0
        self._gpu_worker_state: dict[int, int] = {}
        self._container_restarts_total: dict[int, int] = defaultdict(int)

    def observe_latency(self, task: str, gpu: int, latency_seconds: float):
        key = (task, gpu)
        with self._lock:
            for i, le in enumerate(LATENCY_BUCKETS):
                if latency_seconds <= le:
                    self._latency_buckets[key][i] += 1
                    break
            self._latency_sum[key] += latency_seconds
            self._latency_count[key] += 1

    def inc_evals(self, success: bool = False, error_type: str = None):
        with self._lock:
            if success:
                self._evals_total[("success", "true")] += 1
            elif error_type:
                self._evals_total[("error_type", error_type)] += 1

    def set_queue_depth(self, depth: int):
        with self._lock:
            self._queue_depth = depth

    def set_gpu_state(self, gpu_id: int, state: str):
        state_map = {"healthy": 1, "degraded": 2, "failed": 3}
        with self._lock:
            self._gpu_worker_state[gpu_id] = state_map.get(state, 0)

    def inc_container_restarts(self, gpu_id: int):
        with self._lock:
            self._container_restarts_total[gpu_id] += 1

    def render_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines = []

        with self._lock:
            if self._latency_buckets:
                lines.append("# HELP eval_latency_seconds Histogram of eval latency")
                lines.append("# TYPE eval_latency_seconds histogram")
                for (task, gpu), buckets in sorted(self._latency_buckets.items()):
                    cumulative = 0
                    for i, le in enumerate(LATENCY_BUCKETS):
                        cumulative += buckets[i]
                        le_str = "+Inf" if le == float("inf") else str(le)
                        lines.append(
                            f'eval_latency_seconds_bucket{{task="{task}",gpu="{gpu}",le="{le_str}"}} {cumulative}'
                        )
                    key = (task, gpu)
                    lines.append(
                        f'eval_latency_seconds_sum{{task="{task}",gpu="{gpu}"}} {self._latency_sum[key]}'
                    )
                    lines.append(
                        f'eval_latency_seconds_count{{task="{task}",gpu="{gpu}"}} {self._latency_count[key]}'
                    )

            lines.append("# HELP queue_depth Current queue depth")
            lines.append("# TYPE queue_depth gauge")
            lines.append(f"queue_depth {self._queue_depth}")

            if self._evals_total:
                lines.append("# HELP evals_total Total evaluations")
                lines.append("# TYPE evals_total counter")
                for (label_name, label_value), count in sorted(
                    self._evals_total.items()
                ):
                    lines.append(
                        f'evals_total{{{label_name}="{label_value}"}} {count}'
                    )

            if self._gpu_worker_state:
                lines.append("# HELP gpu_worker_state GPU worker state (1=healthy, 2=degraded, 3=failed)")
                lines.append("# TYPE gpu_worker_state gauge")
                for gpu_id, state in sorted(self._gpu_worker_state.items()):
                    lines.append(
                        f'gpu_worker_state{{gpu="{gpu_id}"}} {state}'
                    )

            if self._container_restarts_total:
                lines.append("# HELP container_restarts_total Container restart count")
                lines.append("# TYPE container_restarts_total counter")
                for gpu_id, count in sorted(
                    self._container_restarts_total.items()
                ):
                    lines.append(
                        f'container_restarts_total{{gpu="{gpu_id}"}} {count}'
                    )

        return "\n".join(lines) + "\n"


metrics = Metrics()
