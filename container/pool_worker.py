#!/usr/bin/env python3
"""Container pool worker — runs inside the GPU container.

Reads JSON requests from stdin, evaluates CUDA/Triton kernels,
writes comprehensive JSON results to stdout. One request at a time.
"""

import io
import json
import linecache
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml


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


_kernel_counter = 0


def compile_kernel(code, task_module):
    """Compile and load the user-submitted kernel code. Returns the module namespace."""
    global _kernel_counter
    _kernel_counter += 1
    source_file = f"/tmp/_kernel_{_kernel_counter}.py"
    linecache.cache[source_file] = (
        len(code),
        None,
        code.splitlines(True),
        source_file,
    )

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    namespace = {"__name__": "__eval__", "__builtins__": __builtins__}

    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
        try:
            exec(compile(code, source_file, "exec"), namespace)
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
    task_dir = f"/workspace/lib/{task_name}"
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


def _is_popcorn_task(task_name):
    """Check if task uses Popcorn evaluation (task.yml + eval.py)."""
    task_dir = Path(f"/workspace/lib/{task_name}")
    return (task_dir / "task.yml").is_file() and (task_dir / "eval.py").is_file()


def _build_test_string(tests):
    """Serialize test/benchmark dicts to semicolon-delimited format."""
    lines = []
    for test in tests:
        kvs = [f"{k}: {v}" for k, v in test.items()]
        lines.append("; ".join(kvs))
    return "\n".join(lines) + "\n"


def _run_with_popcorn_pipe(args, cwd, timeout):
    """Run a subprocess with POPCORN_FD pipe, return parsed result dict."""
    env = os.environ.copy()
    pipe_read, pipe_write = os.pipe()
    env["POPCORN_FD"] = str(pipe_write)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd,
            pass_fds=[pipe_write],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        os.close(pipe_write)
        os.close(pipe_read)
        return None, {
            "stdout": (e.stdout or ""),
            "stderr": (e.stderr or ""),
            "timed_out": True,
        }

    os.close(pipe_write)
    pipe_output = os.fdopen(pipe_read, "r").read()

    result_dict = {}
    for line in pipe_output.splitlines():
        key, _, value = line.partition(":")
        if key.strip() or value.strip():
            result_dict[key.strip()] = value.strip()

    return result_dict, {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def evaluate_popcorn(code, task_name):
    """Evaluate a kernel using the Popcorn protocol (task.yml + eval.py)."""
    task_dir = Path(f"/workspace/lib/{task_name}")

    with open(task_dir / "task.yml") as f:
        task_cfg = yaml.safe_load(f)

    tests = task_cfg.get("tests", [])
    benchmarks = task_cfg.get("benchmarks", [])
    ranking_by = task_cfg.get("ranking_by", "last")
    test_timeout = task_cfg.get("test_timeout", 1200)
    ranked_timeout = task_cfg.get("ranked_timeout", 1200)

    file_specs = task_cfg.get("files", [])
    main_file = task_cfg.get("config", {}).get("main", "eval.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        for file_spec in file_specs:
            name = file_spec["name"]
            source = file_spec["source"]
            if source == "@SUBMISSION@":
                (Path(tmpdir) / name).write_text(code)
            else:
                src_path = task_dir / source
                if src_path.is_file():
                    (Path(tmpdir) / name).write_text(src_path.read_text())

        all_stdout = ""
        all_stderr = ""

        # Phase 1: correctness tests
        test_string = _build_test_string(tests)
        test_file = Path(tmpdir) / "_tests.txt"
        test_file.write_text(test_string)

        result_dict, proc_info = _run_with_popcorn_pipe(
            ["python3", main_file, "test", str(test_file)],
            cwd=tmpdir,
            timeout=test_timeout,
        )

        all_stdout += proc_info.get("stdout", "")
        all_stderr += proc_info.get("stderr", "")

        if result_dict is None:
            return make_error_result(
                "Evaluation timed out during correctness tests",
                "timeout",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        check_result = result_dict.get("check", "fail")
        if check_result != "pass":
            return make_error_result(
                f"Correctness tests failed (check={check_result})",
                "eval_failure",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        # Phase 2: benchmarks (leaderboard mode)
        bench_string = _build_test_string(benchmarks)
        bench_file = Path(tmpdir) / "_benchmarks.txt"
        bench_file.write_text(bench_string)

        bench_result, bench_info = _run_with_popcorn_pipe(
            ["python3", main_file, "leaderboard", str(bench_file)],
            cwd=tmpdir,
            timeout=ranked_timeout,
        )

        all_stdout += bench_info.get("stdout", "")
        all_stderr += bench_info.get("stderr", "")

        if bench_result is None:
            return make_error_result(
                "Evaluation timed out during benchmarks",
                "timeout",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        bench_check = bench_result.get("check", "fail")
        if bench_check != "pass":
            return make_error_result(
                f"Benchmark correctness check failed (check={bench_check})",
                "eval_failure",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        # Compute score from benchmark results
        num_benchmarks = int(bench_result.get("benchmark-count", 0))
        if num_benchmarks == 0:
            return make_error_result(
                "No benchmark results found",
                "eval_failure",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        bench_means_ns = []
        individual_runs = []
        for i in range(num_benchmarks):
            key = f"benchmark.{i}.mean"
            if key not in bench_result:
                continue
            mean_ns = float(bench_result[key])
            bench_means_ns.append(mean_ns)
            individual_runs.append({
                "benchmark_id": i,
                "config": f"benchmark_{i}",
                "time_us": mean_ns / 1000.0,
            })

        if not bench_means_ns:
            return make_error_result(
                "No valid benchmark timings found",
                "eval_failure",
                logs={"stdout": all_stdout, "stderr": all_stderr,
                      "compilation_log": "", "traceback": None},
            )

        # ns -> seconds -> geometric mean -> microseconds
        if ranking_by == "geom":
            means_sec = [ns / 1e9 for ns in bench_means_ns]
            geom_mean_sec = math.exp(sum(math.log(s) for s in means_sec) / len(means_sec))
            score_us = geom_mean_sec * 1e6
        elif ranking_by == "last":
            score_us = bench_means_ns[-1] / 1000.0
        elif ranking_by == "mean":
            means_sec = [ns / 1e9 for ns in bench_means_ns]
            score_us = (sum(means_sec) / len(means_sec)) * 1e6
        else:
            score_us = bench_means_ns[0] / 1000.0

        return {
            "success": True,
            "score_us": score_us,
            "error": None,
            "error_type": None,
            "logs": {
                "stdout": all_stdout,
                "stderr": all_stderr,
                "compilation_log": "",
                "traceback": None,
            },
            "test_results": {
                "passed": len(tests),
                "failed": 0,
                "total": len(tests),
                "first_failure": None,
                "details": [],
            },
            "benchmark_details": {
                "geom_mean_us": score_us,
                "individual_runs": individual_runs,
            },
        }


def evaluate(request):
    """Evaluate a single kernel submission."""
    code = request.get("code", "")
    task_name = request.get("task_name", "")

    if not code:
        return make_error_result("No code provided", "eval_failure")
    if not task_name:
        return make_error_result("No task_name provided", "eval_failure")

    if _is_popcorn_task(task_name):
        return evaluate_popcorn(code, task_name)

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
