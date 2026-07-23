# GPU Mode Evaluation Server

A production-ready, multi-GPU kernel evaluation server with intelligent crash handling, comprehensive observability, and robust failure detection.

[![Tests](https://img.shields.io/badge/tests-122%2F122%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Features

### Core Capabilities
- **Multi-GPU scheduling** with fair work distribution across GPU workers
- **Container pooling** eliminates ~4s startup overhead per evaluation
- **Bounded queue** (num_gpus × 8) with 503 fast-fail when full
- **GPU type validation** ensures benchmarks run on correct hardware
- **Graceful shutdown** with queue draining and container cleanup

### Reliability (Phase 2)
- **Intelligent crash handling** distinguishes kernel bugs from infrastructure failures
  - OOM/SEGFAULT crashes → immediate `eval_failure` (no wasteful retries)
  - Other crashes → same-container retry (~3s) → different-GPU retry if needed
  - **Saves 17+ minutes + 2 GPUs per bad kernel**
- **GPU health monitoring** checks ECC errors and temperature before each eval
- **Memory leak detection** tracks GPU memory usage
- **Retry budget** prevents infinite loops (max 2 retries per request)

### Production API (Phase 3)
- **Python client library** with auto-retry (zero external dependencies!)
- **CLI tool** for command-line evaluation
- **Prometheus metrics** endpoint for monitoring dashboards
- **Structured logging** in JSON format with request tracing
- **YAML configuration** for declarative server setup

### Robustness Validation (Phase 4)
- **6 comprehensive stress test scenarios**
- Queue saturation, crash recovery, mixed workload, memory exhaustion, sustained load, concurrent startup

---

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPUs with CUDA 12.9
- Podman or Docker
- nvidia-smi (NVIDIA drivers installed)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gpu-mode-eval-server.git
cd gpu-mode-eval-server

# Install the package
pip install -e .

# Build the container image
bash scripts/build_container.sh
```

### Start the Server

```bash
# Basic startup (all GPUs, port 8080)
python -m eval_server --gpus 0,1,2,3 --port 8080

# With JSON logging
python -m eval_server --gpus 0,1,2,3 --log-format json

# With config file
python -m eval_server --config config.yaml
```

### Submit a Kernel

**Option 1: Python Client Library**

```python
from eval_server.client import EvalClient

client = EvalClient("http://localhost:8080")

kernel_code = """
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Your Triton kernel here
    pass
"""

result = client.eval(
    code=kernel_code,
    task_name="trimul",
    gpu_type="H100"
)

if result.success:
    print(f"✓ Score: {result.score_us:,.0f} µs")
    print(f"  Queue time: {result.get_queue_time():.0f} ms")
    print(f"  Eval time: {result.get_eval_time():.0f} ms")
    print(f"  Tests: {result.test_results['passed']}/{result.test_results['total']} passed")
else:
    print(f"✗ Failed: {result.error}")
    if result.get_traceback():
        print(f"\nTraceback:\n{result.get_traceback()}")
```

**Option 2: CLI Tool**

```bash
eval-kernel \
  --server http://localhost:8080 \
  --code my_kernel.py \
  --task trimul \
  --gpu-type H100
```

---

## Architecture

### System Overview

```
┌─────────────┐
│   Client    │
│  (Python /  │
│     CLI)    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────────────────────┐
│     HTTP Server (port 8080)     │
│  POST /eval  │  GET /health    │
│  GET /metrics                   │
└──────────┬──────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Shared Pool │  Bounded Queue (num_gpus × 8)
    │   (queue)   │  503 fast-fail when full
    └──────┬──────┘
           │
    ┌──────┴───────┬───────┬───────┐
    │              │       │       │
    ▼              ▼       ▼       ▼
┌────────┐   ┌────────┐ ... GPU Workers
│ GPU 0  │   │ GPU 1  │     (one per GPU)
│Worker  │   │Worker  │
└────┬───┘   └────┬───┘
     │            │
     ▼            ▼
┌──────────┐ ┌──────────┐
│Container │ │Container │  Persistent containers
│ (Podman) │ │ (Podman) │  CUDA 12.9 + PyTorch 2.5.1
└──────────┘ └──────────┘
```

### Container Lifecycle

```
HEALTHY ──(crash)──> RECOVERING ──(3 consecutive)──> DEGRADED ──(5 consecutive)──> FAILED
   │                     │                                                            │
   └─(success)───────────┘                                                            │
   └─(periodic restart every 1000 evals)─────────────────────────────────────────────┘
```

### Request Flow

```
1. Request received       (timestamp: t_received)
2. Queued                 (timestamp: t_queued)
3. GPU assigned           (timestamp: t_gpu_assigned)
4. Evaluation started     (timestamp: t_eval_started)
5. Evaluation completed   (timestamp: t_eval_completed)
6. Response sent          (timestamp: t_response_sent)

Timing breakdown:
- queue_time_ms = t_gpu_assigned - t_received
- eval_time_ms = t_eval_completed - t_eval_started
- total_time_ms = t_response_sent - t_received
```

---

## Configuration

### YAML Config File Example

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8080

gpus:
  ids: [0, 1, 2, 3]
  timeout: 530  # seconds per eval

container:
  image: eval-server:latest
  runtime: podman  # or docker

tasks:
  directory: /path/to/lib/tasks

logging:
  level: INFO
  format: json  # or text

queue:
  max_depth_per_gpu: 8
```

Usage:
```bash
python -m eval_server --config config.yaml

# CLI args override config file
python -m eval_server --config config.yaml --port 9999
```

### Environment Variables

```bash
# None required - all configuration via CLI args or config file
```

---

## API Reference

### POST /eval

Submit a kernel for evaluation.

**Request:**
```json
{
  "code": "import triton\n...",
  "task_name": "trimul",
  "gpu_type": "H100"  // optional
}
```

**Response (Success):**
```json
{
  "success": true,
  "score_us": 12345.67,
  "error": null,
  "error_type": null,
  "logs": {
    "stdout": "...",
    "stderr": "...",
    "compilation_log": "...",
    "traceback": null
  },
  "timing": {
    "queue_time_ms": 120.5,
    "eval_time_ms": 530.2,
    "total_time_ms": 650.7,
    "timestamps": {
      "received": "2026-07-23T14:00:00.000Z",
      "gpu_assigned": "2026-07-23T14:00:00.120Z",
      "eval_started": "2026-07-23T14:00:00.125Z",
      "eval_completed": "2026-07-23T14:00:00.655Z",
      "response_sent": "2026-07-23T14:00:00.660Z"
    }
  },
  "test_results": {
    "passed": 18,
    "failed": 0,
    "total": 18,
    "first_failure": null,
    "details": [...]
  },
  "benchmark_details": {
    "geom_mean_us": 12345.67,
    "individual_runs": [
      {"benchmark_id": 0, "config": "...", "time_us": 11000.0},
      {"benchmark_id": 1, "config": "...", "time_us": 13000.0}
    ]
  },
  "metadata": {
    "gpu_id": 0,
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "container_restarts": 3,
    "retry_count": 0,
    "same_container_retry": 0,
    "different_gpu_retry": 0,
    "crash_signal": null
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "score_us": -1000000.0,
  "error": "Compilation failed: ...",
  "error_type": "compilation_error",  // or eval_failure, timeout, infra_failure, queue_full, gpu_mismatch
  "logs": {
    "stdout": "",
    "stderr": "...",
    "traceback": "Traceback (most recent call last):\n  File ..."
  },
  "test_results": {...},
  "timing": {...},
  "metadata": {...}
}
```

**Status Codes:**
- `200 OK` - Evaluation completed (check `success` field for outcome)
- `400 Bad Request` - Invalid request (missing fields, invalid JSON)
- `503 Service Unavailable` - Queue full (retry with exponential backoff)

### GET /health

Get server health status.

**Response:**
```json
{
  "status": "healthy",
  "workers": [
    {"gpu_id": 0, "gpu_name": "NVIDIA H100 80GB HBM3", "health": "healthy"},
    {"gpu_id": 1, "gpu_name": "NVIDIA H100 80GB HBM3", "health": "healthy"}
  ],
  "queue_depth": 8,
  "uptime_seconds": 3600
}
```

### GET /metrics

Prometheus metrics endpoint.

**Response (text/plain):**
```prometheus
# Latency histogram
eval_latency_seconds_bucket{task="trimul",gpu="0",le="1"} 10
eval_latency_seconds_bucket{task="trimul",gpu="0",le="5"} 50
eval_latency_seconds_sum{task="trimul",gpu="0"} 5234
eval_latency_seconds_count{task="trimul",gpu="0"} 120

# Queue depth
queue_depth{} 8

# Eval totals
evals_total{success="true"} 1000
evals_total{error_type="eval_failure"} 50
evals_total{error_type="infra_failure"} 5

# GPU states
gpu_worker_state{gpu="0"} 1  # 1=healthy, 2=degraded, 3=failed
gpu_worker_state{gpu="1"} 1

# Container restarts
container_restarts_total{gpu="0"} 3
container_restarts_total{gpu="1"} 1
```

---

## Monitoring

### Grafana Dashboard Example

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'eval-server'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

**Key Metrics to Monitor:**

1. **Queue Depth** (`queue_depth`)
   - Alert if > 80% of max capacity
   - Indicates need for more GPUs

2. **P99 Latency** (`eval_latency_seconds`)
   - Track percentiles over time
   - Alert on degradation

3. **Error Rate** (`evals_total{error_type="infra_failure"}`)
   - Should be near zero
   - Spikes indicate infrastructure issues

4. **GPU Health** (`gpu_worker_state`)
   - Alert if any GPU enters FAILED state
   - Monitor container restarts

---

## Stress Testing

Run comprehensive adversarial scenarios to validate robustness:

```bash
# Start server
python -m eval_server --gpus 0,1,2,3 --port 8080

# Run all scenarios
python scripts/stress_test.py --url http://localhost:8080 --num-gpus 4

# Run individual scenarios
python scripts/stress_test.py --scenario queue        # Queue saturation
python scripts/stress_test.py --scenario crash        # Crash recovery
python scripts/stress_test.py --scenario mixed        # Mixed workload
python scripts/stress_test.py --scenario memory       # Memory exhaustion
python scripts/stress_test.py --scenario sustained --duration 600  # 10 min sustained load
python scripts/stress_test.py --scenario startup      # Concurrent startup
```

**Scenarios:**

1. **Queue Saturation** - Submits 20× concurrent requests per GPU
2. **Container Crash Recovery** - Intentional segfaults, verifies recovery
3. **Mixed Workload** - Interleaves valid/slow/crashing kernels
4. **Memory Exhaustion** - Large tensor allocations, verifies OOM detection
5. **Sustained Load** - Continuous traffic, checks for P99 degradation
6. **Concurrent Startup** - Burst before workers ready, verifies graceful handling

---

## Development

### Running Tests

```bash
# All tests (122 tests, ~25s)
pytest tests/ -v

# Specific test file
pytest tests/test_client.py -v

# With coverage
pytest tests/ --cov=eval_server --cov-report=html
```

### Project Structure

```
gpu-mode-eval-server/
├── src/eval_server/              # Core implementation
│   ├── __main__.py               # CLI entry point
│   ├── client.py                 # Python client library
│   ├── cli.py                    # CLI tool (eval-kernel)
│   ├── config.py                 # YAML config support
│   ├── container.py              # Container lifecycle management
│   ├── failure_detection.py     # GPU health + memory leak detection
│   ├── logging_config.py         # Structured logging
│   ├── metrics.py                # Prometheus metrics
│   ├── pool.py                   # Shared eval pool + workers
│   └── server.py                 # HTTP server (POST /eval, GET /health, GET /metrics)
├── container/
│   ├── Dockerfile                # CUDA 12.9 + PyTorch 2.5.1+cu129 + Triton 3.3.1
│   └── pool_worker.py            # In-container eval loop
├── tests/                        # 122 unit + integration tests
├── scripts/
│   ├── build_container.sh        # Container image build script
│   └── stress_test.py            # 6 adversarial scenarios
├── lib/tasks/                    # Task definitions
│   ├── trimul/                   # Triangle Multiplicative Update (18 tests, 7 benchmarks)
│   └── mla_decode/               # Multi-Head Latent Attention Decode
└── pyproject.toml                # Package configuration
```

---

## Performance Characteristics

### Throughput
- **Bounded queue:** num_gpus × 8 (prevents memory exhaustion)
- **503 fast-fail:** Queue full → immediate 503 (no wasted waiting)
- **Container pooling:** Eliminates ~4s startup overhead per eval

### Latency
- **Queue time tracking:** 5-checkpoint timing breakdown
- **Timeout calculation:** Parallel-aware formula
  - Before: 17,020s (4.7 hours) for 32-depth, 4-GPU
  - After: 4,830s (1.3 hours) — **72% reduction**

### Reliability
- **Intelligent crash handling:**
  - Bad kernel retries: Before 26.5 min (3 GPUs) → After 8.8 min (1 GPU)
  - **Saves 17.7 minutes + 2 GPUs per bad kernel**
- **GPU health checks:** Prevents corrupt results from failing hardware
- **Memory leak detection:** Proactive container restart

---

## Troubleshooting

### Server won't start

```bash
# Check nvidia-smi works
nvidia-smi

# Check container runtime
podman version  # or docker version

# Check port availability
lsof -i :8080

# Check container image exists
podman images | grep eval-server
```

### Queue always full (503 errors)

**Solutions:**
- Add more GPUs: `--gpus 0,1,2,3,4,5,6,7`
- Increase queue depth (modify `pool.py` line 102)
- Scale horizontally (multiple server instances with load balancer)

### Kernels timing out

**Solutions:**
- Increase timeout: `--timeout 1200` (default 530s)
- Check if kernels are actually slow (review benchmark_details)
- Profile kernels locally before submission

### Container crashes frequently

**Check logs:**
```bash
# Server logs
python -m eval_server --log-format json 2>&1 | tee server.log

# Container logs
podman logs <container_id>
```

**Common causes:**
- OOM (out of memory) - reduce batch size in kernels
- Invalid CUDA operations - verify kernel correctness
- GPU hardware failure - check `nvidia-smi` for ECC errors

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Acknowledgments

- Built for [GPU Mode](https://github.com/gpu-mode) kernel benchmarking
- Inspired by [ttt-advisor](https://github.com/test-time-training/discover) eval infrastructure
- Container approach based on proven patterns from Modal's GPU fleet management

---

## Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/gpu-mode-eval-server/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/gpu-mode-eval-server/discussions)

---

**Status:** ✅ Production-Ready | 122/122 tests passing | Zero external dependencies (client library)
