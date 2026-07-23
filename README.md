# GPU Mode Evaluation Server

Production-ready multi-GPU kernel evaluation server with intelligent crash handling, comprehensive observability, and robust failure detection.

[![Tests](https://img.shields.io/badge/tests-122%2F122%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)

## Features

- **Multi-GPU scheduling** — fair work distribution with bounded queue (num_gpus × 8)
- **Container pooling** — eliminates ~4s startup overhead per evaluation
- **Intelligent crash handling** — distinguishes kernel bugs from infrastructure failures
  - OOM/SEGFAULT crashes → immediate `eval_failure` (no wasteful retries)
  - Saves 17+ minutes + 2 GPUs per bad kernel
- **GPU health monitoring** — checks ECC errors and temperature
- **Python client library** — zero external dependencies, auto-retry on failures
- **Prometheus metrics** — latency histograms, queue depth, error rates
- **Structured logging** — JSON format with request tracing

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/ash-ding/gpu-mode-eval-server.git
cd gpu-mode-eval-server
pip install -e .

# Build container image
bash scripts/build_container.sh
```

### Start Server

```bash
# Basic
python -m eval_server --gpus 0,1,2,3 --port 8080

# With JSON logging (recommended for production)
python -m eval_server --gpus 0,1,2,3 --log-format json

# With config file
python -m eval_server --config config.yaml
```

### Use Client Library

```python
from eval_server.client import EvalClient

client = EvalClient("http://localhost:8080")

kernel_code = """
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Your optimized Triton kernel here
    pass
"""

result = client.eval(code=kernel_code, task_name="trimul", gpu_type="H100")

if result.success:
    print(f"✓ Score: {result.score_us:,.0f} µs")
    print(f"  Queue time: {result.get_queue_time():.0f} ms")
    print(f"  Tests: {result.test_results['passed']}/{result.test_results['total']} passed")
else:
    print(f"✗ Error: {result.error}")
```

### Use CLI Tool

```bash
eval-kernel \
  --server http://localhost:8080 \
  --code my_kernel.py \
  --task trimul \
  --gpu-type H100
```

---

## Adding New Benchmark Tasks

The eval server auto-discovers tasks in `lib/tasks/`. To add a new benchmark:

### 1. Create Task Directory

```bash
mkdir lib/tasks/my_new_task
cd lib/tasks/my_new_task
```

### 2. Create `task.yml` Configuration

```yaml
# task.yml
files:
  - {"name": "submission.py", "source": "@SUBMISSION@"}
  - {"name": "task.py", "source": "task.py"}
  - {"name": "reference.py", "source": "reference.py"}
  - {"name": "eval.py", "source": "eval.py"}

lang: "py"
description: "Brief description of what this benchmark measures"

config:
  main: "eval.py"

templates:
  Python: "submission.py"

test_timeout: 1200
benchmark_timeout: 1200
ranking_by: "geom"

tests:
  - {"param1": 32, "param2": 128, "seed": 1234}
  - {"param1": 64, "param2": 256, "seed": 5678}
  # Add 10-20 test cases

benchmarks:
  - {"param1": 256, "param2": 512, "seed": 1234}
  - {"param1": 512, "param2": 1024, "seed": 5678}
  # Add 5-10 benchmark cases
```

### 3. Implement Task Interface (`task.py`)

```python
import torch

class Task:
    def __init__(self, **config):
        self.param1 = config["param1"]
        self.param2 = config["param2"]
        self.seed = config.get("seed", 0)
        
    def prepare_data(self) -> tuple:
        """Generate input data for this test case."""
        torch.manual_seed(self.seed)
        input_tensor = torch.randn(self.param1, self.param2, device="cuda")
        return (input_tensor,)
    
    def check_output(self, output, reference_output) -> bool:
        """Verify output correctness."""
        return torch.allclose(output, reference_output, rtol=1e-3, atol=1e-5)
```

### 4. Implement Reference Solution (`reference.py`)

```python
import torch

def reference_solution(input_tensor):
    """Reference implementation (correctness over performance)."""
    # Ground truth implementation here
    return input_tensor * 2  # Example
```

### 5. Implement Evaluation Harness (`eval.py`)

```python
import torch
from task import Task
from reference import reference_solution

def run_tests(submission_fn, tests):
    """Run correctness tests."""
    passed = 0
    failed_tests = []
    
    for i, test_config in enumerate(tests):
        task = Task(**test_config)
        inputs = task.prepare_data()
        
        try:
            output = submission_fn(*inputs)
            reference = reference_solution(*inputs)
            if task.check_output(output, reference):
                passed += 1
            else:
                failed_tests.append({"test_id": i, "error": "Output mismatch"})
        except Exception as e:
            failed_tests.append({"test_id": i, "error": str(e)})
    
    return {
        "passed": passed,
        "failed": len(failed_tests),
        "total": len(tests),
        "first_failure": failed_tests[0] if failed_tests else None,
        "details": failed_tests,
    }

def run_benchmarks(submission_fn, benchmarks):
    """Run performance benchmarks."""
    results = []
    
    for i, bench_config in enumerate(benchmarks):
        task = Task(**bench_config)
        inputs = task.prepare_data()
        
        # Warmup
        for _ in range(3):
            submission_fn(*inputs)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            submission_fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            
            times.append(start.elapsed_time(end) * 1000)  # µs
        
        results.append({
            "benchmark_id": i,
            "config": bench_config,
            "time_us": sorted(times)[len(times) // 2],  # median
        })
    
    return results
```

### 6. Create Submission Template (`submission.py`)

```python
import torch

def kernel(input_tensor):
    """
    User implements this function.
    
    Args:
        input_tensor: Input of shape [param1, param2]
    
    Returns:
        Output tensor of same shape
    """
    raise NotImplementedError("Implement your optimized kernel here")
```

### 7. Restart Server

```bash
# Server auto-discovers the new task
python -m eval_server --gpus 0,1,2,3 --tasks-dir ./lib/tasks
```

### 8. Test Your Task

```python
client = EvalClient("http://localhost:8080")
result = client.eval(code=my_code, task_name="my_new_task")
```

**That's it!** No code changes to the server needed. See `lib/tasks/trimul/` for a complete example.

---

## Configuration

### YAML Config File

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8080

gpus:
  ids: [0, 1, 2, 3]
  timeout: 530

container:
  image: eval-server:latest
  runtime: podman

tasks:
  directory: /path/to/lib/tasks

logging:
  level: INFO
  format: json
```

Usage:
```bash
python -m eval_server --config config.yaml

# CLI args override config file
python -m eval_server --config config.yaml --port 9999
```

---

## Server-Side Logging

### Log Formats

**Text (default):**
```bash
python -m eval_server --gpus 0,1,2,3
```

**JSON (recommended for production):**
```bash
python -m eval_server --gpus 0,1,2,3 --log-format json
```

### What Gets Logged

Every request is fully traced with a unique `request_id`:

**Successful eval:**
```
INFO  HTTP request received: request_id=req_000001 task=trimul code_hash=a1b2c3d4 client=127.0.0.1
INFO  Request received: request_id=req_000001 task=trimul queue_depth=3
INFO  Request assigned: request_id=req_000001 gpu_id=0 queue_time_ms=120.5
INFO  Eval completed: request_id=req_000001 gpu_id=0 success=True score_us=12345.0 
      duration_ms=530.2 tests_passed=18/18
```

**Failed eval:**
```
ERROR Compilation failed: request_id=req_000002 task=trimul gpu_id=0 
      error=SyntaxError: invalid syntax at line 10
```

**Container crash:**
```
ERROR Container exited with SIGKILL/OOM (code -9) for GPU 0, task=trimul
WARNING Kernel bug detected: gpu_id=0 signal=SIGKILL/OOM - marking as eval_failure (no retry)
```

### Log Analysis

```bash
# Find specific request
grep "req_000123" server.log

# Count successful evals today
grep "success=True" server.log | wc -l

# Find all compilation errors (JSON format)
jq 'select(.message | contains("Compilation failed"))' server.jsonl

# Find slow evals (>60s)
jq 'select(.message | contains("Slow eval detected"))' server.jsonl

# Count requests by task
jq -r 'select(.task) | .task' server.jsonl | sort | uniq -c | sort -rn
```

### Response Payload

Every eval returns comprehensive audit information:

```python
result = client.eval(code=kernel_code, task_name="trimul")

# Full response structure
result.response = {
    "success": True,
    "score_us": 12345.0,
    "error": None,
    "error_type": None,
    "logs": {
        "stdout": "...",
        "stderr": "...",
        "compilation_log": "...",
        "traceback": None
    },
    "timing": {
        "queue_time_ms": 120.5,
        "eval_time_ms": 530.2,
        "total_time_ms": 650.7,
        "timestamps": {
            "received": "2026-07-23T14:00:00.000Z",
            "queued": "2026-07-23T14:00:00.005Z",
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
        "first_failure": None,
        "details": [...]
    },
    "benchmark_details": {
        "geom_mean_us": 12345.67,
        "individual_runs": [...]
    },
    "metadata": {
        "request_id": "req_000001",
        "gpu_id": 0,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "container_restarts": 3,
        "retry_count": 0,
        "same_container_retry": 0,
        "different_gpu_retry": 0,
        "crash_signal": None
    }
}
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
  "gpu_type": "H100"
}
```

**Response:**
- `200 OK` — Evaluation completed (check `success` field)
- `400 Bad Request` — Invalid request
- `503 Service Unavailable` — Queue full (retry with exponential backoff)

### GET /health

Server health status.

**Response:**
```json
{
  "status": "healthy",
  "workers": [
    {"gpu_id": 0, "gpu_name": "NVIDIA H100 80GB HBM3", "health": "healthy"}
  ],
  "queue_depth": 8,
  "uptime_seconds": 3600
}
```

### GET /metrics

Prometheus metrics endpoint (text format).

**Key metrics:**
- `eval_latency_seconds` — Latency histogram by task/GPU
- `queue_depth` — Current queue depth
- `evals_total` — Eval counter by success/error_type
- `gpu_worker_state` — GPU health (1=healthy, 2=degraded, 3=failed)
- `container_restarts_total` — Container restart count per GPU

---

## Monitoring

### Prometheus Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'eval-server'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics to Monitor

1. **Queue Depth** (`queue_depth`)
   - Alert if > 80% of max capacity
   - Indicates need for more GPUs

2. **P99 Latency** (`eval_latency_seconds`)
   - Track percentiles over time
   - Alert on degradation

3. **Error Rate** (`evals_total{error_type="..."}`)
   - Should be near zero for `infra_failure`
   - Track `compilation_error` and `eval_failure` trends

4. **GPU Health** (`gpu_worker_state`)
   - Alert if any GPU enters FAILED state
   - Monitor container restarts

---

## Troubleshooting

### Server Won't Start

**Check prerequisites:**
```bash
# Verify nvidia-smi works
nvidia-smi

# Check container runtime
podman version  # or: docker version

# Check port availability
lsof -i :8080

# Verify container image exists
podman images | grep eval-server
```

**Solution:** Build container image if missing:
```bash
bash scripts/build_container.sh
```

### Queue Always Full (503 Errors)

**Symptoms:** Clients receive `503 Service Unavailable` frequently

**Diagnosis:**
```bash
# Check queue depth warnings in logs
grep "Queue filling up" server.log
```

**Solutions:**
1. Add more GPUs: `--gpus 0,1,2,3,4,5,6,7`
2. Increase timeout (if evals are legitimately slow): `--timeout 1200`
3. Scale horizontally: run multiple server instances with a load balancer

### Kernels Timing Out

**Symptoms:** Evals return `error_type: "timeout"`

**Diagnosis:**
```bash
# Check eval durations in logs
grep "Eval completed" server.log | grep duration_ms

# Find slow evals
grep "Slow eval detected" server.log
```

**Solutions:**
1. Increase timeout: `--timeout 1200` (default 530s)
2. Profile kernels locally before submission
3. Check if kernels are actually slow (review `benchmark_details` in response)

### Container Crashes Frequently

**Symptoms:** Logs show repeated container crashes

**Diagnosis:**
```bash
# Check crash logs
grep "Container exited" server.log

# Check for OOM crashes
grep "SIGKILL/OOM" server.log

# Check container logs directly
podman logs <container_id>
```

**Common causes:**
- **OOM (out of memory)** — Reduce batch size or tensor sizes in kernel
- **Segmentation fault** — Fix memory access bugs in kernel code
- **GPU hardware failure** — Run `nvidia-smi` to check for ECC errors

**Solutions:**
```bash
# Check GPU health
nvidia-smi

# Check GPU memory usage
nvidia-smi dmon -s um

# Check for ECC errors
nvidia-smi -q | grep -A 10 "ECC Errors"
```

### High Latency / Slow Performance

**Symptoms:** Queue times or eval times higher than expected

**Diagnosis:**
```bash
# Check queue depth over time
grep "queue_depth" server.log

# Check P99 latency
jq 'select(.message | contains("Eval completed")) | .duration_ms' server.jsonl | \
  sort -n | tail -10

# Check for slow evals
grep "Slow eval detected" server.log
```

**Solutions:**
1. **High queue time** → Add more GPUs or reduce request rate
2. **High eval time** → Profile kernels, check for GPU contention
3. **Container restarts** → Check logs for crash causes

### Debugging a Stuck Request

**Symptoms:** Client request never returns

**Diagnosis:**
```bash
# Find the request in logs (use request_id from client metadata)
grep "req_000123" server.log

# Check which stage it reached:
# - "HTTP request received" but no "Request received" → stuck in HTTP layer
# - "Request received" but no "Request assigned" → stuck in queue
# - "Request assigned" but no "Eval completed" → stuck in eval
```

**Solutions:**
1. **Stuck in queue** → Queue is full, server is overloaded
2. **Stuck in eval** → Check container logs, may need to increase timeout
3. **No logs at all** → Check client is hitting correct server URL

### Viewing Logs in Real-Time

```bash
# Text format (human-readable)
python -m eval_server --gpus 0,1,2,3 2>&1 | tee server.log

# JSON format (machine-parseable)
python -m eval_server --gpus 0,1,2,3 --log-format json 2>&1 | tee server.jsonl

# Follow logs in real-time
tail -f server.log

# Filter for errors only
tail -f server.jsonl | jq 'select(.level == "ERROR")'
```

---

## Project Structure

```
gpu-mode-eval-server/
├── src/eval_server/          # Core implementation
│   ├── __main__.py            # CLI entry point
│   ├── client.py              # Python client library
│   ├── cli.py                 # CLI tool (eval-kernel)
│   ├── config.py              # YAML config support
│   ├── container.py           # Container lifecycle management
│   ├── failure_detection.py  # GPU health + memory leak detection
│   ├── logging_config.py      # Structured logging
│   ├── metrics.py             # Prometheus metrics
│   ├── pool.py                # Eval pool + workers
│   └── server.py              # HTTP server
├── container/
│   ├── Dockerfile             # CUDA 12.9 + PyTorch 2.5.1 + Triton 3.3.1
│   └── pool_worker.py         # In-container eval loop
├── lib/tasks/                 # Task definitions
│   ├── trimul/                # Example: Triangle Multiplicative Update
│   └── mla_decode/            # Example: Multi-Head Latent Attention
├── scripts/
│   ├── build_container.sh     # Container build script
│   └── stress_test.py         # 6 adversarial test scenarios
└── tests/                     # 122 unit + integration tests
```

---

## Performance Characteristics

- **Throughput:** Bounded queue (num_gpus × 8) prevents memory exhaustion
- **Latency:** Container pooling eliminates ~4s startup overhead
- **Reliability:** Intelligent crash handling saves 17+ minutes per bad kernel
- **Timeout:** Parallel-aware calculation (72% faster than naive approach)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/`
5. Open a Pull Request

---

## Support

- **Issues:** [GitHub Issues](https://github.com/ash-ding/gpu-mode-eval-server/issues)
- **Repository:** https://github.com/ash-ding/gpu-mode-eval-server

---

**Status:** ✅ Production-Ready | 122/122 tests passing | Zero external dependencies (client library)
