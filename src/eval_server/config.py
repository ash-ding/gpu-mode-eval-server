"""YAML configuration file support."""

import yaml


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_config_with_args(config: dict, args) -> dict:
    """Merge config file with CLI args (CLI args take precedence over defaults)."""
    server = config.get("server", {})
    gpus_config = config.get("gpus", {})
    container = config.get("container", {})
    tasks = config.get("tasks", {})
    logging_config = config.get("logging", {})

    merged = {}

    merged["host"] = (
        args.host if args.host != "0.0.0.0" else server.get("host", "0.0.0.0")
    )
    merged["port"] = (
        args.port if args.port != 8080 else server.get("port", 8080)
    )

    if args.gpus != "0":
        merged["gpus"] = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        merged["gpus"] = gpus_config.get("ids", [0])

    merged["timeout"] = (
        args.timeout if args.timeout != 530 else gpus_config.get("timeout", 530)
    )

    merged["image"] = (
        args.image
        if args.image != "eval-server:latest"
        else container.get("image", "eval-server:latest")
    )

    merged["tasks_dir"] = args.tasks_dir or tasks.get("directory")

    merged["log_level"] = (
        args.log_level
        if args.log_level != "INFO"
        else logging_config.get("level", "INFO")
    )
    merged["log_format"] = (
        args.log_format
        if args.log_format != "text"
        else logging_config.get("format", "text")
    )

    return merged
