"""CLI tool for evaluating kernels via the eval server."""

import argparse
import sys
from pathlib import Path

from .client import EvalClient


def main():
    parser = argparse.ArgumentParser(description="Evaluate a kernel via eval server")
    parser.add_argument("--server", required=True, help="Server URL")
    parser.add_argument("--code", required=True, help="Path to kernel code file")
    parser.add_argument(
        "--task", required=True, help="Task name (trimul, mla_decode)"
    )
    parser.add_argument("--gpu-type", help="GPU type (H100, A100, etc.)")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    code_path = Path(args.code)
    if not code_path.exists():
        print(f"Error: file not found: {args.code}", file=sys.stderr)
        sys.exit(1)

    code = code_path.read_text()
    client = EvalClient(args.server, timeout=args.timeout)

    print(f"Submitting kernel to {args.server}...")
    try:
        result = client.eval(code=code, task_name=args.task, gpu_type=args.gpu_type)
    except ConnectionError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        sys.exit(1)

    if result.success:
        print(f"Success: {result.score_us:,.0f} us")
        print(f"Queue time: {result.get_queue_time():.0f} ms")
        print(f"Eval time: {result.get_eval_time():.0f} ms")
        tr = result.test_results
        print(f"Tests: {tr.get('passed', 0)}/{tr.get('total', 0)} passed")
    else:
        print(f"Failed: {result.error}")
        print(f"Error type: {result.error_type}")
        if result.get_traceback():
            print(f"\nTraceback:\n{result.get_traceback()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
