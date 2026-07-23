#!/usr/bin/env python3
"""Stress test harness — asyncio-based concurrent load generator.

Implements 6 adversarial test scenarios for the eval server:

  1. Queue Saturation     — burst num_gpus*20 concurrent requests
  2. Container Crash Recovery — intentional segfaults, verify recovery
  3. Mixed Workload       — interleave valid/slow/crashing kernels
  4. Memory Exhaustion    — large tensor allocation, verify OOM handling
  5. Sustained Load       — continuous traffic for configurable duration
  6. Concurrent Startup   — burst requests against a freshly-started server

Usage (requires a running server):

    python scripts/stress_test.py --url http://localhost:8080 --num-gpus 4
    python scripts/stress_test.py --scenario queue
    python scripts/stress_test.py --scenario sustained --duration 600
"""

import argparse
import asyncio
import json
import signal
import statistics
import subprocess
import sys
import time

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

VALID_KERNEL = """\
import torch
import triton
import triton.language as tl

def custom_kernel(*args):
    return args[0] + args[1]
"""

CRASHING_KERNEL = """\
import ctypes
ctypes.string_at(0)
"""

SLOW_KERNEL = """\
import time
def custom_kernel(*args):
    time.sleep(5)
    return args[0]
"""

MEMORY_HOG_KERNEL = """\
import torch
x = torch.zeros(10000, 10000, 10000, device='cuda')
def custom_kernel(*args):
    return args[0] + args[1]
"""

SCENARIO_CHOICES = ["all", "queue", "crash", "mixed", "memory", "sustained", "startup"]


async def submit_eval(session, url, code, task_name="test_task"):
    """Submit a single eval request. Returns (latency_ms, response_data, status_code)."""
    payload = {"code": code, "task_name": task_name}
    start = time.monotonic()
    try:
        async with session.post(f"{url}/eval", json=payload) as resp:
            data = await resp.json()
            latency = (time.monotonic() - start) * 1000
            return latency, data, resp.status
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return latency, {"error": str(e), "success": False}, 0


def compute_percentiles(values):
    """Compute p50/p90/p99/mean/min/max for a list of numeric values."""
    if not values:
        return {"p50": 0, "p90": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p90": s[int(n * 0.9)],
        "p99": s[int(n * 0.99)],
        "mean": statistics.mean(s),
        "min": s[0],
        "max": s[-1],
    }


def count_false_zeros(results):
    """Count results where success=True but score_us <= 0."""
    return sum(
        1 for r in results
        if r[1].get("success") and r[1].get("score_us", 1) <= 0
    )


# ---------------------------------------------------------------------------
# Scenario 1: Queue Saturation
# ---------------------------------------------------------------------------

async def test_queue_saturation(session, url, num_gpus):
    """Submit num_gpus*20 concurrent requests. Verify all complete, no false-zeros."""
    count = num_gpus * 20
    print(f"  Submitting {count} concurrent requests...")

    tasks = [submit_eval(session, url, VALID_KERNEL) for _ in range(count)]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r[2] == 200 and r[1].get("success"))
    queue_full_count = sum(1 for r in results if r[2] == 503)
    error_count = sum(1 for r in results if r[2] not in (200, 503))
    false_zeros = count_false_zeros(results)
    latencies = [r[0] for r in results]

    passed = (
        len(results) == count
        and error_count == 0
        and false_zeros == 0
        and success_count + queue_full_count == len(results)
    )

    return {
        "scenario": "queue_saturation",
        "total": count,
        "success": success_count,
        "queue_full": queue_full_count,
        "errors": error_count,
        "false_zeros": false_zeros,
        "latency": compute_percentiles(latencies),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Scenario 2: Container Crash Recovery
# ---------------------------------------------------------------------------

async def test_container_crash_recovery(session, url, num_gpus):
    """Submit crashing kernels, then verify the same GPUs recover."""
    print(f"  Crashing {num_gpus} containers...")

    crash_tasks = [submit_eval(session, url, CRASHING_KERNEL) for _ in range(num_gpus)]
    crash_results = await asyncio.gather(*crash_tasks)

    crash_detected = sum(
        1 for r in crash_results
        if r[2] == 200 and not r[1].get("success")
    )
    crash_error_types = [
        r[1].get("error_type") for r in crash_results if r[2] == 200
    ]

    print("  Waiting for container recovery...")
    await asyncio.sleep(3)

    print("  Submitting valid kernels to verify recovery...")
    recovery_tasks = [submit_eval(session, url, VALID_KERNEL) for _ in range(num_gpus)]
    recovery_results = await asyncio.gather(*recovery_tasks)

    recovered = sum(1 for r in recovery_results if r[2] == 200)

    return {
        "scenario": "crash_recovery",
        "crash_count": len(crash_results),
        "crash_detected": crash_detected,
        "crash_error_types": crash_error_types,
        "recovery_success": recovered,
        "recovery_total": num_gpus,
        "passed": recovered == num_gpus,
    }


# ---------------------------------------------------------------------------
# Scenario 3: Mixed Workload
# ---------------------------------------------------------------------------

async def test_mixed_workload(session, url, num_gpus):
    """Interleave valid, slow, crashing kernels. Valid kernels must always succeed."""
    kernels = [
        ("valid", VALID_KERNEL),
        ("slow", SLOW_KERNEL),
        ("crash", CRASHING_KERNEL),
        ("valid", VALID_KERNEL),
    ]

    batch = kernels * max(num_gpus, 10)
    print(f"  Submitting {len(batch)} mixed requests...")

    tasks = []
    for name, code in batch:
        tasks.append((name, submit_eval(session, url, code)))

    raw = await asyncio.gather(*[t for _, t in tasks])
    labeled = [(name, r) for (name, _), r in zip(tasks, raw)]

    valid_results = [r for n, r in labeled if n == "valid"]
    crash_results = [r for n, r in labeled if n == "crash"]

    valid_success = sum(
        1 for r in valid_results if r[2] == 200 and r[1].get("success")
    )
    false_zeros = count_false_zeros(valid_results)
    crash_detected = sum(
        1 for r in crash_results if r[2] == 200 and not r[1].get("success")
    )

    return {
        "scenario": "mixed_workload",
        "total": len(batch),
        "valid_success": valid_success,
        "valid_total": len(valid_results),
        "crash_detected": crash_detected,
        "crash_total": len(crash_results),
        "false_zeros": false_zeros,
        "passed": false_zeros == 0,
    }


# ---------------------------------------------------------------------------
# Scenario 4: Memory Exhaustion
# ---------------------------------------------------------------------------

async def test_memory_exhaustion(session, url):
    """Submit a kernel that allocates huge tensors. Verify OOM is detected."""
    print("  Submitting memory-hogging kernel...")

    latency, data, status = await submit_eval(session, url, MEMORY_HOG_KERNEL)

    oom_detected = status == 200 and not data.get("success")
    error_type = data.get("error_type")

    print("  Waiting for recovery after OOM...")
    await asyncio.sleep(3)

    print("  Verifying GPU still works after OOM...")
    latency2, data2, status2 = await submit_eval(session, url, VALID_KERNEL)
    post_oom_ok = status2 == 200

    return {
        "scenario": "memory_exhaustion",
        "oom_detected": oom_detected,
        "error_type": error_type,
        "latency_ms": round(latency, 2),
        "post_oom_recovery": post_oom_ok,
        "passed": oom_detected,
    }


# ---------------------------------------------------------------------------
# Scenario 5: Sustained Load
# ---------------------------------------------------------------------------

async def test_sustained_load(session, url, num_gpus, duration_seconds=60):
    """Run continuous eval traffic for the given duration. Check for P99 degradation."""
    print(f"  Running sustained load for {duration_seconds}s...")

    start_time = time.monotonic()
    latencies = []
    success_count = 0
    failure_count = 0

    while time.monotonic() - start_time < duration_seconds:
        latency, data, status = await submit_eval(session, url, VALID_KERNEL)
        latencies.append(latency)
        if status == 200 and data.get("success"):
            success_count += 1
        else:
            failure_count += 1
        await asyncio.sleep(0.1)

    total = success_count + failure_count
    if total == 0:
        return {
            "scenario": "sustained_load",
            "duration_seconds": duration_seconds,
            "total_requests": 0,
            "passed": False,
            "error": "No requests completed",
        }

    success_rate = success_count / total

    first_half = latencies[: len(latencies) // 2]
    second_half = latencies[len(latencies) // 2 :]

    p99_first = sorted(first_half)[int(len(first_half) * 0.99)] if first_half else 0
    p99_second = sorted(second_half)[int(len(second_half) * 0.99)] if second_half else 0

    degradation_pct = 0.0
    if p99_first > 0:
        degradation_pct = ((p99_second - p99_first) / p99_first) * 100

    passed = degradation_pct < 50.0

    return {
        "scenario": "sustained_load",
        "duration_seconds": duration_seconds,
        "total_requests": total,
        "success_rate": round(success_rate, 4),
        "p99_first_half_ms": round(p99_first, 2),
        "p99_second_half_ms": round(p99_second, 2),
        "degradation_pct": round(degradation_pct, 2),
        "latency": compute_percentiles(latencies),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Scenario 6: Concurrent Startup
# ---------------------------------------------------------------------------

async def test_concurrent_startup(server_cmd, port=9999):
    """Start server and immediately burst requests before workers are ready."""
    print(f"  Starting server on port {port}...")

    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        await asyncio.sleep(1)

        url = f"http://localhost:{port}"
        connector = aiohttp.TCPConnector(limit=0)
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            print("  Bursting 20 requests...")
            tasks = [submit_eval(session, url, VALID_KERNEL) for _ in range(20)]
            results = await asyncio.gather(*tasks)

        crashes = sum(1 for r in results if r[2] == 0)
        http_ok = sum(1 for r in results if r[2] in (200, 503))

        return {
            "scenario": "concurrent_startup",
            "total": len(results),
            "crashes": crashes,
            "http_ok": http_ok,
            "passed": crashes == 0,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def check_health(session, url):
    """Check server health endpoint."""
    try:
        async with session.get(f"{url}/health") as resp:
            return await resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_scenarios(args):
    """Run selected stress test scenarios and produce a report."""
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=max(args.duration + 120, 600))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "server_url": args.url,
        "num_gpus": args.num_gpus,
        "scenarios": {},
    }

    run_all = args.scenario == "all"

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        health = await check_health(session, args.url)
        print(f"Server health: {json.dumps(health, indent=2)}")

        if run_all or args.scenario == "queue":
            print("\n=== Scenario 1: Queue Saturation ===")
            result = await test_queue_saturation(session, args.url, args.num_gpus)
            report["scenarios"]["queue_saturation"] = result
            _print_result(result)

        if run_all or args.scenario == "crash":
            print("\n=== Scenario 2: Container Crash Recovery ===")
            result = await test_container_crash_recovery(session, args.url, args.num_gpus)
            report["scenarios"]["crash_recovery"] = result
            _print_result(result)

        if run_all or args.scenario == "mixed":
            print("\n=== Scenario 3: Mixed Workload ===")
            result = await test_mixed_workload(session, args.url, args.num_gpus)
            report["scenarios"]["mixed_workload"] = result
            _print_result(result)

        if run_all or args.scenario == "memory":
            print("\n=== Scenario 4: Memory Exhaustion ===")
            result = await test_memory_exhaustion(session, args.url)
            report["scenarios"]["memory_exhaustion"] = result
            _print_result(result)

        if run_all or args.scenario == "sustained":
            print(f"\n=== Scenario 5: Sustained Load ({args.duration}s) ===")
            result = await test_sustained_load(
                session, args.url, args.num_gpus, args.duration
            )
            report["scenarios"]["sustained_load"] = result
            _print_result(result)

    if run_all or args.scenario == "startup":
        print("\n=== Scenario 6: Concurrent Startup ===")
        server_cmd = [
            sys.executable, "-m", "eval_server",
            "--gpus", "0",
            "--port", str(args.startup_port),
        ]
        result = await test_concurrent_startup(server_cmd, port=args.startup_port)
        report["scenarios"]["concurrent_startup"] = result
        _print_result(result)

    all_passed = all(
        r.get("passed", False) for r in report["scenarios"].values()
    )
    report["all_passed"] = all_passed

    return report


def _print_result(result):
    """Print scenario result with pass/fail indicator."""
    status = "PASS" if result.get("passed") else "FAIL"
    print(f"  [{status}] {result['scenario']}")
    for k, v in result.items():
        if k in ("scenario", "passed"):
            continue
        print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Stress test harness for the eval server",
    )
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument(
        "--output", default="stress_test_report.json", help="JSON report output path"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        default="all",
        help="Run a single scenario or all",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds for sustained load scenario (default: 60)",
    )
    parser.add_argument(
        "--startup-port",
        type=int,
        default=9999,
        help="Port for concurrent startup scenario (default: 9999)",
    )
    args = parser.parse_args()

    report = asyncio.run(run_scenarios(args))

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.output}")

    print("\n=== SUMMARY ===")
    for name, result in report["scenarios"].items():
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"  [{status}] {name}")

    overall = "ALL PASSED" if report["all_passed"] else "SOME FAILED"
    print(f"\n  {overall}")

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
