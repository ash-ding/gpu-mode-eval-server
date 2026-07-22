"""HTTP server — POST /eval and GET /health endpoints."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .metrics import metrics
from .pool import EvalRequest, SharedEvalPool

logger = logging.getLogger(__name__)


class EvalHandler(BaseHTTPRequestHandler):
    pool: SharedEvalPool
    request_timeout: int

    def do_POST(self):
        if self.path != "/eval":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        code = data.get("code", "")
        task_name = data.get("task_name", "")
        gpu_type = data.get("gpu_type")

        if not code:
            self._send_json(400, {"error": "Missing 'code' field"})
            return
        if not task_name:
            self._send_json(400, {"error": "Missing 'task_name' field"})
            return

        if gpu_type is not None:
            matching = self.pool.get_matching_gpus(gpu_type)
            if not matching:
                available = ", ".join(
                    f"{k}: {v}" for k, v in self.pool.gpu_names.items()
                )
                self._send_json(200, {
                    "success": False,
                    "score_us": -1_000_000.0,
                    "error": f"Requested GPU type '{gpu_type}' not available. Available: {available}",
                    "error_type": "gpu_mismatch",
                    "logs": {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
                    "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
                    "benchmark_details": None,
                    "timing": None,
                    "metadata": None,
                })
                return

        request = EvalRequest(code, task_name, gpu_type)

        if not self.pool.submit(request):
            self._send_json(503, {
                "error": "Queue full, try again later",
                "error_type": "queue_full",
                "queue_depth": self.pool.queue_depth(),
            })
            return

        done = request.done.wait(timeout=self.request_timeout)

        if not done:
            self._send_json(504, {
                "error": "Request timed out waiting for GPU",
                "error_type": "timeout",
            })
            return

        self._send_json(200, request.result)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, self.pool.get_health())
        elif self.path == "/metrics":
            body = metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        logger.debug("%s - %s", self.address_string(), format % args)


def create_server(
    pool: SharedEvalPool,
    host: str = "0.0.0.0",
    port: int = 8080,
    request_timeout: int = 600,
    thread_pool_size: int = 512,
) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (EvalHandler,), {
        "pool": pool,
        "request_timeout": request_timeout,
    })

    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.request_queue_size = thread_pool_size

    logger.info("Server listening on %s:%d", host, port)
    return server
