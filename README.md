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

### Adding New Benchmark Tasks

The eval server uses a task-based architecture that makes it easy to add new GPU kernel benchmarks. Each task is self-contained in its own directory under `lib/tasks/`.

#### Task Structure

Each task directory must contain:

```
lib/tasks/my_new_task/
├── task.yml          # Task configuration (required)
├── task.py           # Task interface (required)
├── eval.py           # Evaluation harness (required)
├── reference.py      # Reference implementation (required)
├── submission.py     # Submission template (required)
└── utils.py          # Helper functions (optional)
```

#### Step-by-Step Guide

**1. Create Task Directory**

```bash
mkdir lib/tasks/my_new_task
cd lib/tasks/my_new_task
```

**2. Write `task.yml` Configuration**

```yaml
# lib/tasks/my_new_task/task.yml

files:
  - {"name": "submission.py", "source": "@SUBMISSION@"}
  - {"name": "task.py", "source": "task.py"}
  - {"name": "utils.py", "source": "utils.py"}
  - {"name": "reference.py", "source": "reference.py"}
  - {"name": "eval.py", "source": "eval.py"}

lang: "py"

description: |
  Brief description of what this benchmark measures.
  
  Input format:
  - Describe inputs here
  
  Output format:
  - Describe expected outputs here

config:
  main: "eval.py"

templates:
  Python: "submission.py"

test_timeout: 1200       # Max seconds for all tests
benchmark_timeout: 1200  # Max seconds for all benchmarks
ranked_timeout: 1200     # Max seconds for ranking
ranking_by: "geom"       # "geom" (geometric mean) or "sum"

# Test cases (for correctness verification)
tests:
  - {"param1": 32, "param2": 128, "seed": 1234}
  - {"param1": 64, "param2": 256, "seed": 5678}
  # Add 10-20 test cases covering edge cases

# Benchmark cases (for performance measurement)
benchmarks:
  - {"param1": 256, "param2": 512, "seed": 1234}
  - {"param1": 512, "param2": 1024, "seed": 5678}
  # Add 5-10 benchmark cases representing realistic workloads
```

**3. Implement `task.py` Interface**

```python
# lib/tasks/my_new_task/task.py

import torch

class Task:
    """Task interface for my_new_task."""
    
    def __init__(self, **config):
        """Initialize task with test/benchmark config."""
        self.param1 = config["param1"]
        self.param2 = config["param2"]
        self.seed = config.get("seed", 0)
        
    def prepare_data(self) -> tuple:
        """Generate input data for this test case."""
        torch.manual_seed(self.seed)
        # Generate inputs based on self.param1, self.param2, etc.
        input_tensor = torch.randn(self.param1, self.param2, device="cuda")
        return (input_tensor,)
    
    def check_output(self, output, reference_output) -> bool:
        """Verify output correctness."""
        return torch.allclose(output, reference_output, rtol=1e-3, atol=1e-5)
```

**4. Implement `reference.py` Ground Truth**

```python
# lib/tasks/my_new_task/reference.py

import torch

def reference_solution(input_tensor):
    """Reference implementation (doesn't need to be fast)."""
    # Implement the correct algorithm here
    # This is the ground truth for correctness checking
    result = input_tensor * 2  # Example
    return result
```

**5. Implement `eval.py` Harness**

```python
# lib/tasks/my_new_task/eval.py

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
        
        # Run submission
        try:
            output = submission_fn(*inputs)
        except Exception as e:
            failed_tests.append({"test_id": i, "error": str(e)})
            continue
        
        # Check correctness
        reference_output = reference_solution(*inputs)
        if task.check_output(output, reference_output):
            passed += 1
        else:
            failed_tests.append({"test_id": i, "error": "Output mismatch"})
    
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
        
        # Benchmark (10 runs)
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            submission_fn(*inputs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # Convert to microseconds
        
        median_time = sorted(times)[len(times) // 2]
        results.append({
            "benchmark_id": i,
            "config": bench_config,
            "time_us": median_time,
        })
    
    return results
```

**6. Create `submission.py` Template**

```python
# lib/tasks/my_new_task/submission.py

import torch

def kernel(input_tensor):
    """
    User implements this function.
    
    Args:
        input_tensor: Input tensor of shape [param1, param2]
    
    Returns:
        Output tensor of same shape
    """
    # TODO: Implement optimized kernel here
    raise NotImplementedError("Implement this kernel")
```

**7. Update Server to Load New Task**

The server auto-discovers tasks in `lib/tasks/` — no code changes needed! Just restart the server:

```bash
python -m eval_server --gpus 0,1,2,3 --tasks-dir ./lib/tasks
```

**8. Test Your New Task**

```python
from eval_server.client import EvalClient

client = EvalClient("http://localhost:8080")

kernel_code = """
import torch

def kernel(input_tensor):
    return input_tensor * 2  # Your optimized implementation
"""

result = client.eval(
    code=kernel_code,
    task_name="my_new_task",  # Your task name (directory name)
    gpu_type="H100"
)

print(f"Tests: {result.test_results['passed']}/{result.test_results['total']}")
print(f"Score: {result.score_us} µs")
```

#### Task Design Best Practices

1. **Test Coverage**: Include 10-20 test cases covering:
   - Small inputs (fast tests)
   - Edge cases (empty, single element, max size)
   - Different data distributions
   - Masked vs unmasked scenarios

2. **Benchmark Selection**: Choose 5-10 benchmarks representing:
   - Realistic workload sizes
   - Range of input shapes
   - Performance-critical configurations

3. **Timeout Tuning**: Set timeouts based on expected runtime:
   - Fast kernels (<10s): `test_timeout: 600` (10 min)
   - Slow kernels (30s-2min): `test_timeout: 1200` (20 min)
   - Very slow kernels: `test_timeout: 3600` (1 hour)

4. **Ranking Metric**:
   - `ranking_by: "geom"` — geometric mean (default, handles wide range of times)
   - `ranking_by: "sum"` — sum of all benchmark times

5. **Reference Implementation**: Prioritize correctness over performance. Use PyTorch built-ins when possible.

---

## Server-Side Logging and Audit Trail

The eval server maintains comprehensive logs of all evaluation requests and results, useful for debugging, monitoring, and audit purposes.

### Log Formats

The server supports two log formats via `--log-format`:

**Text Format (Default)**
```
2026-07-23 15:30:45 INFO Starting eval server with GPUs: [0, 1, 2, 3]
2026-07-23 15:30:46 INFO GPU 0: NVIDIA H100 80GB HBM3
2026-07-23 15:30:47 INFO Eval pool started: 4 GPUs, queue depth 32
```

**JSON Format (Recommended for Production)**
```bash
python -m eval_server --log-format json
```

```json
{"timestamp": "2026-07-23T15:30:45.123Z", "level": "INFO", "message": "Starting eval server with GPUs: [0, 1, 2, 3]", "logger": "eval_server"}
{"timestamp": "2026-07-23T15:30:46.456Z", "level": "INFO", "message": "GPU 0: NVIDIA H100 80GB HBM3", "logger": "eval_server"}
{"timestamp": "2026-07-23T15:30:47.789Z", "level": "INFO", "message": "Eval pool started: 4 GPUs, queue depth 32", "logger": "eval_server.pool"}
```

### What Gets Logged

**1. Server Lifecycle Events**
- Server startup/shutdown
- GPU detection and worker initialization
- Container lifecycle (start, restart, health status changes)

**2. Request Processing**
- Queue depth warnings (when queue is filling up)
- GPU type mismatch warnings
- Retry attempts (same-container and different-GPU retries)
- Infrastructure failures

**3. Error Conditions**
- Container crashes with signal information
- Evaluation timeouts
- Queue saturation events
- GPU health check failures

### Accessing Logs

**Real-time Log Streaming**
```bash
# Text format (human-readable)
python -m eval_server --gpus 0,1,2,3 2>&1 | tee server.log

# JSON format (machine-parseable)
python -m eval_server --gpus 0,1,2,3 --log-format json 2>&1 | tee server.jsonl
```

**Filtering JSON Logs**
```bash
# Find all errors
jq 'select(.level == "ERROR")' server.jsonl

# Find all retry attempts
jq 'select(.message | contains("requeuing"))' server.jsonl

# Find all evals for a specific task
jq 'select(.task_name == "trimul")' server.jsonl

# Track a specific request by ID
jq 'select(.request_id == "req_abc123")' server.jsonl
```

### Response Payload — Complete Audit Trail

Every evaluation returns a comprehensive response with full audit information:

```python
result = client.eval(code=kernel_code, task_name="trimul")

# All information is in the response
print(result.response)
```

**Response Fields:**

1. **Outcome**
   - `success`: True/False
   - `error`: Error message (if failed)
   - `error_type`: Category (compilation_error, eval_failure, timeout, etc.)

2. **Performance Metrics**
   - `score_us`: Geometric mean benchmark time in microseconds
   - `benchmark_details`: Individual benchmark runs with times

3. **Logs and Debug Info**
   - `logs.stdout`: Standard output from kernel execution
   - `logs.stderr`: Standard error output
   - `logs.compilation_log`: Triton compilation output
   - `logs.traceback`: Full Python traceback (if error)

4. **Test Results**
   - `test_results.passed`: Number of tests passed
   - `test_results.failed`: Number of tests failed
   - `test_results.total`: Total test count
   - `test_results.first_failure`: Details of first failing test
   - `test_results.details`: Full list of all test results

5. **Timing Breakdown** (5 checkpoints)
   - `timing.queue_time_ms`: Time spent waiting in queue
   - `timing.eval_time_ms`: Time spent in evaluation
   - `timing.total_time_ms`: End-to-end latency
   - `timing.timestamps`: All 6 lifecycle timestamps
     - `received`: Request received by server
     - `queued`: Request added to queue
     - `gpu_assigned`: GPU worker picked up request
     - `eval_started`: Evaluation began
     - `eval_completed`: Evaluation finished
     - `response_sent`: Response sent to client

6. **Infrastructure Metadata**
   - `metadata.gpu_id`: Which GPU handled this request
   - `metadata.gpu_name`: GPU model name (e.g., "NVIDIA H100 80GB HBM3")
   - `metadata.container_restarts`: How many times container restarted
   - `metadata.retry_count`: Total retry attempts
   - `metadata.same_container_retry`: Same-container retry count
   - `metadata.different_gpu_retry`: Different-GPU retry count
   - `metadata.crash_signal`: Crash signal if container crashed (e.g., "SIGKILL/OOM")

### Persistent Audit Trail

**Option 1: Save Responses to Database**

```python
import sqlite3
import json
from datetime import datetime

db = sqlite3.connect("eval_audit.db")
db.execute("""
    CREATE TABLE IF NOT EXISTS evals (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        task_name TEXT,
        success BOOLEAN,
        score_us REAL,
        gpu_id INTEGER,
        queue_time_ms REAL,
        eval_time_ms REAL,
        error_type TEXT,
        full_response TEXT
    )
""")

# After each eval
result = client.eval(code=kernel_code, task_name="trimul")
db.execute("""
    INSERT INTO evals (timestamp, task_name, success, score_us, gpu_id, 
                       queue_time_ms, eval_time_ms, error_type, full_response)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    datetime.utcnow().isoformat(),
    "trimul",
    result.success,
    result.score_us,
    result.response["metadata"]["gpu_id"],
    result.get_queue_time(),
    result.get_eval_time(),
    result.error_type,
    json.dumps(result.response),
))
db.commit()
```

**Option 2: Append to JSONL File**

```python
import json
from datetime import datetime

def log_eval(result, task_name, kernel_id):
    """Append eval result to audit log."""
    with open("eval_audit.jsonl", "a") as f:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_name": task_name,
            "kernel_id": kernel_id,
            "result": result.response,
        }
        f.write(json.dumps(record) + "\n")

# Usage
result = client.eval(code=kernel_code, task_name="trimul")
log_eval(result, "trimul", kernel_id="v1.2.3")
```

**Option 3: Server-Side Logging to File**

```bash
# Redirect all server logs to file with timestamps
python -m eval_server --log-format json 2>&1 | \
  jq -c '. + {written_at: now | todate}' >> /var/log/eval-server.jsonl
```

### Log Retention and Analysis

**Daily Log Rotation**
```bash
# Run server with logrotate
python -m eval_server --log-format json 2>&1 | \
  rotatelogs /var/log/eval-server-%Y-%m-%d.jsonl 86400
```

**Query Historical Data**
```bash
# How many evals per day last week?
cat /var/log/eval-server-2026-07-*.jsonl | \
  jq -s 'group_by(.timestamp[:10]) | map({date: .[0].timestamp[:10], count: length})'

# What's the P95 queue time?
cat /var/log/eval-server-today.jsonl | \
  jq -s 'map(.queue_time_ms) | sort | .[length * 0.95 | floor]'
```

### Integration with Monitoring Systems

**Prometheus Metrics** (already built-in via `/metrics`)
- Real-time metrics for queue depth, latency, error rates
- See [Monitoring](#monitoring) section above

**Grafana Dashboards**
- Visualize queue depth over time
- Track P50/P95/P99 latency
- Alert on error rate spikes

**ELK Stack** (Elasticsearch, Logstash, Kibana)
```bash
# Ship JSON logs to Logstash
python -m eval_server --log-format json 2>&1 | \
  nc logstash-server 5000
```

**CloudWatch / Datadog**
```python
# Add instrumentation to client code
import datadog

result = client.eval(code=kernel_code, task_name="trimul")
datadog.statsd.histogram("eval.latency", result.get_eval_time(), tags=[f"task:{task_name}"])
datadog.statsd.increment("eval.count", tags=[f"success:{result.success}"])
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
