"""CLI entry point: python -m eval_server"""

import argparse
import atexit
import logging
import os
import signal
import sys

from .config import load_config, merge_config_with_args
from .container import ContainerRuntime, PooledKernelEvaluator, query_gpu_names
from .logging_config import setup_logging
from .pool import SharedEvalPool
from .server import create_server


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU kernel evaluation server",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=530,
        help="Per-eval timeout in seconds (default: 530)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="eval-server:latest",
        help="Container image name (default: eval-server:latest)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Path to lib/tasks/ directory to mount into containers",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Log format (default: text)",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=None,
        help="Max eval queue depth (default: num_gpus * 8)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        file_config = load_config(args.config)
        settings = merge_config_with_args(file_config, args)
    else:
        settings = {
            "host": args.host,
            "port": args.port,
            "gpus": [int(g.strip()) for g in args.gpus.split(",")],
            "timeout": args.timeout,
            "image": args.image,
            "tasks_dir": args.tasks_dir,
            "log_level": args.log_level,
            "log_format": args.log_format,
            "max_queue_depth": args.max_queue_depth,
        }

    setup_logging(level=settings["log_level"], log_format=settings["log_format"])

    gpu_ids = settings["gpus"]
    logging.info("Starting eval server with GPUs: %s", gpu_ids)

    gpu_names = query_gpu_names(gpu_ids)
    if gpu_names:
        for gid, name in gpu_names.items():
            logging.info("GPU %d: %s", gid, name)
    else:
        logging.warning("Could not query GPU names — GPU type validation disabled")

    runtime = ContainerRuntime()

    tasks_dir = settings["tasks_dir"]
    if tasks_dir:
        tasks_dir = os.path.abspath(tasks_dir)

    evaluators = [
        PooledKernelEvaluator(
            gpu_id=gid,
            runtime=runtime,
            image=settings["image"],
            timeout=settings["timeout"],
            tasks_dir=tasks_dir,
        )
        for gid in gpu_ids
    ]

    num_gpus = len(gpu_ids)
    max_queue_depth = settings.get("max_queue_depth") or num_gpus * 8
    request_timeout = (max_queue_depth // num_gpus + 1) * settings["timeout"] + 60
    thread_pool_size = max(num_gpus * 128, 512)

    pool = SharedEvalPool(
        evaluators, gpu_names,
        eval_timeout=settings["timeout"],
        max_queue_depth=max_queue_depth,
    )
    pool.start()

    server = create_server(
        pool,
        host=settings["host"],
        port=settings["port"],
        request_timeout=request_timeout,
        thread_pool_size=thread_pool_size,
    )

    def shutdown(*_):
        logging.info("Shutting down...")
        server.shutdown()
        pool.stop()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    atexit.register(pool.stop)

    logging.info(
        "Eval server ready: %d GPUs, port %d, timeout %ds, queue depth %d",
        num_gpus, settings["port"], settings["timeout"], max_queue_depth,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
