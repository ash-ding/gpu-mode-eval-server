#!/usr/bin/env python3
"""Container pool worker — runs inside the GPU container.

Reads JSON requests from stdin, evaluates CUDA/Triton kernels,
writes comprehensive JSON results to stdout. One request at a time.
"""

import io
import json
import math
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout


def make_error_result(error_msg, error_type="eval_failure", logs=None):
    return {
        "success": False,
        "score_us": -1_000_000.0,
        "error": error_msg,
        "error_type": error_type,
        "logs": logs or {"stdout": "", "stderr": "", "compilation_log": "", "traceback": None},
        "test_results": {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []},
        "benchmark_details": None,
    }


def compile_kernel(code, task_module):
    """Compile and load the user-submitted kernel code. Returns the module namespace."""
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    namespace = {"__name__": "__eval__", "__builtins__": __builtins__}

    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        try:
            exec(compile(code, "<submitted_kernel>", "exec"), namespace)
        except Exception:
            tb = traceback.format_exc()
            return None, {
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "compilation_log": captured_stderr.getvalue(),
                "traceback": tb,
            }

    return namespace, {
        "stdout": captured_stdout.getvalue(),
        "stderr": captured_stderr.getvalue(),
        "compilation_log": captured_stderr.getvalue(),
        "traceback": None,
    }


def run_correctness_tests(namespace, task_module):
    """Run ALL correctness tests — never stop on first failure."""
    import torch

    test_cases = task_module.get("test_cases", [])
    if not test_cases:
        return {"passed": 0, "failed": 0, "total": 0, "first_failure": None, "details": []}

    custom_kernel = namespace.get("custom_kernel")
    if custom_kernel is None:
        return {
            "passed": 0,
            "failed": len(test_cases),
            "total": len(test_cases),
            "first_failure": {
                "test_id": 0,
                "name": test_cases[0].get("name", "test_0"),
                "error": "No 'custom_kernel' function found in submitted code",
            },
            "details": [
                {
                    "test_id": i,
                    "name": tc.get("name", f"test_{i}"),
                    "passed": False,
                    "error": "No 'custom_kernel' function found in submitted code",
                }
                for i, tc in enumerate(test_cases)
            ],
        }

    results = []
    first_failure = None
    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases):
        test_name = tc.get("name", f"test_{i}")
        try:
            inputs = tc["setup"]()
            expected = tc["reference"](*inputs)
            actual = custom_kernel(*inputs)

            if isinstance(expected, torch.Tensor):
                rtol = tc.get("rtol", 1e-5)
                atol = tc.get("atol", 1e-8)
                ok = torch.allclose(actual, expected, rtol=rtol, atol=atol)
                if not ok:
                    max_diff = (actual - expected).abs().max().item()
                    error_msg = f"Tensors not close: max_diff={max_diff:.6e}, rtol={rtol}, atol={atol}"
                else:
                    error_msg = None
            else:
                ok = actual == expected
                error_msg = None if ok else f"Expected {expected}, got {actual}"

            if ok:
                passed += 1
                results.append({"test_id": i, "name": test_name, "passed": True, "error": None})
            else:
                failed += 1
                results.append({"test_id": i, "name": test_name, "passed": False, "error": error_msg})
                if first_failure is None:
                    first_failure = {"test_id": i, "name": test_name, "error": error_msg}

        except Exception:
            failed += 1
            tb = traceback.format_exc()
            results.append({"test_id": i, "name": test_name, "passed": False, "error": tb})
            if first_failure is None:
                first_failure = {"test_id": i, "name": test_name, "error": tb}

    return {
        "passed": passed,
        "failed": failed,
        "total": len(test_cases),
        "first_failure": first_failure,
        "details": results,
    }


def run_benchmarks(namespace, task_module):
    """Run benchmarks if all tests passed. Returns benchmark details."""
    import torch

    custom_kernel = namespace.get("custom_kernel")
    if custom_kernel is None:
        return None

    benchmark_configs = task_module.get("benchmark_configs", [])
    if not benchmark_configs:
        return None

    individual_runs = []
    times = []

    for i, config in enumerate(benchmark_configs):
        config_name = config.get("name", f"config_{i}")
        try:
            inputs = config["setup"]()

            # Warmup
            for _ in range(3):
                custom_kernel(*inputs)
            torch.cuda.synchronize()

            # Timed runs
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

            for j in range(10):
                start_events[j].record()
                custom_kernel(*inputs)
                end_events[j].record()

            torch.cuda.synchronize()

            run_times = [start_events[j].elapsed_time(end_events[j]) * 1000 for j in range(10)]
            median_us = sorted(run_times)[5]
            times.append(median_us)

            individual_runs.append({
                "benchmark_id": i,
                "config": config_name,
                "time_us": median_us,
            })

        except Exception:
            individual_runs.append({
                "benchmark_id": i,
                "config": config_name,
                "time_us": -1.0,
            })

    if times:
        geom_mean = math.exp(sum(math.log(t) for t in times if t > 0) / max(len([t for t in times if t > 0]), 1))
    else:
        geom_mean = -1.0

    return {
        "geom_mean_us": geom_mean,
        "individual_runs": individual_runs,
    }


def load_task(task_name):
    """Load task definition from /workspace/lib/tasks/<task_name>/."""
    task_dir = f"/workspace/lib/tasks/{task_name}"
    task_file = os.path.join(task_dir, "task.py")

    if not os.path.isfile(task_file):
        return None, f"Task '{task_name}' not found at {task_file}"

    namespace = {}
    try:
        with open(task_file) as f:
            exec(compile(f.read(), task_file, "exec"), namespace)
    except Exception:
        return None, f"Failed to load task '{task_name}': {traceback.format_exc()}"

    return namespace, None


def evaluate(request):
    """Evaluate a single kernel submission."""
    code = request.get("code", "")
    task_name = request.get("task_name", "")

    if not code:
        return make_error_result("No code provided", "eval_failure")
    if not task_name:
        return make_error_result("No task_name provided", "eval_failure")

    kernel_ns, compile_logs = compile_kernel(code, {})
    if kernel_ns is None:
        return make_error_result(
            "Compilation failed",
            "compilation_error",
            logs=compile_logs,
        )

    task_module, err = load_task(task_name)
    if task_module is None:
        return make_error_result(err, "eval_failure")

    all_stdout = compile_logs["stdout"]
    all_stderr = compile_logs["stderr"]
    all_traceback = compile_logs["traceback"]

    test_stdout = io.StringIO()
    test_stderr = io.StringIO()

    with redirect_stdout(test_stdout), redirect_stderr(test_stderr):
        try:
            test_results = run_correctness_tests(kernel_ns, task_module)
        except Exception:
            tb = traceback.format_exc()
            return make_error_result(
                f"Test runner crashed: {tb}",
                "eval_failure",
                logs={
                    "stdout": all_stdout + test_stdout.getvalue(),
                    "stderr": all_stderr + test_stderr.getvalue(),
                    "compilation_log": compile_logs["compilation_log"],
                    "traceback": tb,
                },
            )

    all_stdout += test_stdout.getvalue()
    all_stderr += test_stderr.getvalue()

    benchmark_details = None
    score_us = -1_000_000.0

    if test_results["failed"] == 0 and test_results["total"] > 0:
        bench_stdout = io.StringIO()
        bench_stderr = io.StringIO()

        with redirect_stdout(bench_stdout), redirect_stderr(bench_stderr):
            try:
                benchmark_details = run_benchmarks(kernel_ns, task_module)
            except Exception:
                tb = traceback.format_exc()
                all_traceback = tb

        all_stdout += bench_stdout.getvalue()
        all_stderr += bench_stderr.getvalue()

        if benchmark_details and benchmark_details["geom_mean_us"] > 0:
            score_us = benchmark_details["geom_mean_us"]

    success = test_results["failed"] == 0 and test_results["total"] > 0

    return {
        "success": success,
        "score_us": score_us if success else -1_000_000.0,
        "error": None if success else f"{test_results['failed']}/{test_results['total']} tests failed",
        "error_type": None if success else "eval_failure",
        "logs": {
            "stdout": all_stdout,
            "stderr": all_stderr,
            "compilation_log": compile_logs["compilation_log"],
            "traceback": all_traceback,
        },
        "test_results": test_results,
        "benchmark_details": benchmark_details,
    }


def main():
    """Main loop: read JSON from stdin, evaluate, write JSON to stdout."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
        except json.JSONDecodeError as e:
            result = make_error_result(f"Invalid JSON input: {e}", "eval_failure")
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            continue

        try:
            result = evaluate(request)
        except Exception:
            tb = traceback.format_exc()
            result = make_error_result(
                f"Unexpected error: {tb}",
                "eval_failure",
                logs={"stdout": "", "stderr": "", "compilation_log": "", "traceback": tb},
            )

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
