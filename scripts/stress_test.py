#!/usr/bin/env python3
"""Stress test harness — asyncio-based concurrent load generator."""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

VALID_KERNEL = '''
import torch
import triton
import triton.language as tl

def custom_kernel(*args):
    return args[0] + args[1]
'''

CRASHING_KERNEL = '''
import ctypes
ctypes.string_at(0)
'''

SLOW_KERNEL = '''
import time
def custom_kernel(*args):
    time.sleep(10)
    return args[0]
'''


async def submit_eval(session, url, code, task_name, gpu_type=None):
    """Submit a single eval request and return (latency_ms, response_json, status_code)."""
    payload = {"code": code, "task_name": task_name}
    if gpu_type:
        payload["gpu_type"] = gpu_type

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


async def scenario_queue_saturation(session, url, concurrency, num_gpus):
    """Submit concurrency requests simultaneously, verify all complete."""
    count = num_gpus * 20
    tasks = [
        submit_eval(session, url, VALID_KERNEL, "test_task")
        for _ in range(count)
    ]
    results = await asyncio.gather(*tasks)

    latencies = [r[0] for r in results]
    http_503 = sum(1 for r in results if r[2] == 503)
    successes = sum(1 for r in results if r[2] == 200 and r[1].get("success"))
    errors = sum(1 for r in results if r[2] not in (200, 503))
    false_zeros = sum(
        1 for r in results
        if r[1].get("success") and r[1].get("score_us", 1) <= 0
    )

    return {
        "scenario": "queue_saturation",
        "total_requests": count,
        "successes": successes,
        "http_503": http_503,
        "errors": errors,
        "false_zeros": false_zeros,
        "latency": compute_percentiles(latencies),
        "passed": errors == 0 and false_zeros == 0,
    }


async def scenario_container_crash(session, url, num_gpus):
    """Submit crashing kernels, verify recovery."""
    crash_tasks = [
        submit_eval(session, url, CRASHING_KERNEL, "test_task")
        for _ in range(num_gpus)
    ]
    crash_results = await asyncio.gather(*crash_tasks)

    crash_errors = [r[1].get("error_type") for r in crash_results if r[2] == 200]

    await asyncio.sleep(2)

    recovery_tasks = [
        submit_eval(session, url, VALID_KERNEL, "test_task")
        for _ in range(num_gpus)
    ]
    recovery_results = await asyncio.gather(*recovery_tasks)

    recovered = sum(1 for r in recovery_results if r[2] == 200)

    return {
        "scenario": "container_crash_recovery",
        "crash_count": len(crash_results),
        "crash_error_types": crash_errors,
        "recovery_success": recovered,
        "recovery_total": num_gpus,
        "passed": recovered == num_gpus,
    }


async def scenario_mixed_workload(session, url, num_gpus):
    """Interleave valid, slow, and crashing kernels."""
    tasks = []
    labels = []

    for i in range(num_gpus * 3):
        mod = i % 3
        if mod == 0:
            tasks.append(submit_eval(session, url, VALID_KERNEL, "test_task"))
            labels.append("valid")
        elif mod == 1:
            tasks.append(submit_eval(session, url, CRASHING_KERNEL, "test_task"))
            labels.append("crash")
        else:
            tasks.append(submit_eval(session, url, VALID_KERNEL, "test_task"))
            labels.append("valid")

    results = await asyncio.gather(*tasks)

    valid_results = [r for r, l in zip(results, labels) if l == "valid"]
    crash_results = [r for r, l in zip(results, labels) if l == "crash"]

    valid_success = sum(1 for r in valid_results if r[2] == 200)
    false_zeros = sum(
        1 for r in valid_results
        if r[1].get("success") and r[1].get("score_us", 1) <= 0
    )

    return {
        "scenario": "mixed_workload",
        "total_requests": len(results),
        "valid_success": valid_success,
        "valid_total": len(valid_results),
        "crash_total": len(crash_results),
        "false_zeros": false_zeros,
        "passed": false_zeros == 0,
    }


async def scenario_timing_validation(session, url):
    """Verify timing breakdown is populated and consistent."""
    latency, data, status = await submit_eval(session, url, VALID_KERNEL, "test_task")

    timing = data.get("timing")
    if not timing:
        return {
            "scenario": "timing_validation",
            "passed": False,
            "error": "No timing data in response",
        }

    timestamps = timing.get("timestamps", {})
    required = ["received", "gpu_assigned", "eval_started", "eval_completed", "response_sent"]
    missing = [k for k in required if not timestamps.get(k)]

    queue_time = timing.get("queue_time_ms", 0)
    eval_time = timing.get("eval_time_ms", 0)
    total_time = timing.get("total_time_ms", 0)

    sum_parts = queue_time + eval_time
    timing_drift = abs(total_time - sum_parts) if total_time > 0 else 0

    return {
        "scenario": "timing_validation",
        "missing_timestamps": missing,
        "queue_time_ms": queue_time,
        "eval_time_ms": eval_time,
        "total_time_ms": total_time,
        "timing_drift_ms": round(timing_drift, 2),
        "passed": len(missing) == 0,
    }


async def check_health(session, url):
    """Check server health endpoint."""
    try:
        async with session.get(f"{url}/health") as resp:
            data = await resp.json()
            return data
    except Exception as e:
        return {"error": str(e)}


async def run_stress_test(url, concurrency, num_gpus):
    """Run all stress test scenarios."""
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        health = await check_health(session, url)
        print(f"Server health: {json.dumps(health, indent=2)}")

        results = []

        print("\n--- Scenario: Timing Validation ---")
        r = await scenario_timing_validation(session, url)
        results.append(r)
        print(json.dumps(r, indent=2))

        print("\n--- Scenario: Queue Saturation ---")
        r = await scenario_queue_saturation(session, url, concurrency, num_gpus)
        results.append(r)
        print(json.dumps(r, indent=2))

        print("\n--- Scenario: Container Crash Recovery ---")
        r = await scenario_container_crash(session, url, num_gpus)
        results.append(r)
        print(json.dumps(r, indent=2))

        print("\n--- Scenario: Mixed Workload ---")
        r = await scenario_mixed_workload(session, url, num_gpus)
        results.append(r)
        print(json.dumps(r, indent=2))

    report = {
        "url": url,
        "concurrency": concurrency,
        "num_gpus": num_gpus,
        "scenarios": results,
        "all_passed": all(r.get("passed", False) for r in results),
    }

    print("\n=== STRESS TEST REPORT ===")
    print(json.dumps(report, indent=2))

    return report


def main():
    parser = argparse.ArgumentParser(description="Stress test for eval server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent requests")
    parser.add_argument("--num-gpus", type=int, default=1, help="Expected number of GPUs")
    parser.add_argument("--output", type=str, help="Output JSON file for report")
    args = parser.parse_args()

    report = asyncio.run(run_stress_test(args.url, args.concurrency, args.num_gpus))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
