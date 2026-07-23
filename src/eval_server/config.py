"""YAML configuration file support."""

import sys

import yaml


def load_config(path: str) -> dict:
    """Load YAML config file."""
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
            return config if config is not None else {}
    except FileNotFoundError:
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file {path}: {e}", file=sys.stderr)
        sys.exit(1)


def merge_config_with_args(config: dict, args) -> dict:
    """Merge config file with CLI args (CLI args always take precedence)."""
    gpus_config = config.get("gpus", {})
    tasks = config.get("tasks", {})
    logging_config = config.get("logging", {})

    merged = {}

    merged["host"] = args.host
    merged["port"] = args.port

    if args.gpus:
        merged["gpus"] = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        merged["gpus"] = gpus_config.get("ids", [0])

    merged["timeout"] = args.timeout
    merged["image"] = args.image
    merged["tasks_dir"] = args.tasks_dir or tasks.get("directory")
    merged["log_level"] = args.log_level
    merged["log_format"] = args.log_format

    return merged
