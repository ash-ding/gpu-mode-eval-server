"""Python client for the eval server with auto-retry and batch support."""

import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json


class EvalResult:
    """Rich result object with accessor methods."""

    def __init__(self, data: dict):
        self._data = data
        self.success = data.get("success", False)
        self.score_us = data.get("score_us", -1_000_000.0)
        self.error = data.get("error")
        self.error_type = data.get("error_type")
        self.logs = data.get("logs", {})
        self.timing = data.get("timing", {})
        self.test_results = data.get("test_results", {})
        self.benchmark_details = data.get("benchmark_details")
        self.metadata = data.get("metadata", {})

    def get_traceback(self) -> Optional[str]:
        """Extract Python traceback if present."""
        return self.logs.get("traceback")

    def get_queue_time(self) -> float:
        """Get queue wait time in milliseconds."""
        return self.timing.get("queue_time_ms", 0.0)

    def get_eval_time(self) -> float:
        """Get eval execution time in milliseconds."""
        return self.timing.get("eval_time_ms", 0.0)

    def get_failed_tests(self) -> List[Dict]:
        """Get list of failed test cases."""
        details = self.test_results.get("details", [])
        return [t for t in details if not t.get("passed", False)]

    def __repr__(self) -> str:
        if self.success:
            return f"EvalResult(success=True, score_us={self.score_us})"
        return f"EvalResult(success=False, error_type={self.error_type!r})"


class EvalClient:
    """Client for eval server with auto-retry and batch support.

    Uses urllib from the standard library — no external dependencies required.
    """

    def __init__(self, base_url: str, timeout: int = 600, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def eval(
        self,
        code: str,
        task_name: str,
        gpu_type: Optional[str] = None,
    ) -> EvalResult:
        """Evaluate a kernel with automatic retry on 503/connection errors."""
        payload: Dict[str, Any] = {
            "code": code,
            "task_name": task_name,
        }
        if gpu_type:
            payload["gpu_type"] = gpu_type

        data = json.dumps(payload).encode("utf-8")

        for attempt in range(self.max_retries):
            try:
                req = Request(
                    f"{self.base_url}/eval",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return EvalResult(body)

            except HTTPError as e:
                if e.code == 503 and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                if e.code == 503:
                    raise ConnectionError(
                        f"Server returned 503 after {self.max_retries} retries"
                    ) from e
                body = json.loads(e.read().decode("utf-8"))
                return EvalResult(body)

            except URLError as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise ConnectionError(
                    f"Connection failed after {self.max_retries} retries: {e}"
                ) from e

        raise RuntimeError(f"Failed after {self.max_retries} retries")

    def eval_batch(self, items: List[Dict[str, Any]]) -> List[EvalResult]:
        """Batch evaluation — submit all sequentially, return results."""
        return [self.eval(**item) for item in items]

    def health(self) -> dict:
        """Get server health status."""
        req = Request(f"{self.base_url}/health", method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            raise ConnectionError(f"Health check failed: HTTP {e.code}") from e
        except URLError as e:
            raise ConnectionError(f"Health check failed: {e}") from e
