"""Tests for config file loading and CLI arg merging."""

import os
import tempfile
import unittest

from eval_server.config import load_config, merge_config_with_args


class TestLoadConfig(unittest.TestCase):
    """Test YAML config file loading."""

    def test_load_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("server:\n  host: 0.0.0.0\n  port: 9090\n")
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        self.assertEqual(config["server"]["port"], 9090)

    def test_load_empty_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("")
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)
        self.assertEqual(config, {})

    def test_load_full_config(self):
        yaml_content = """
server:
  host: 0.0.0.0
  port: 9090
gpus:
  ids: [0, 1, 2, 3]
  timeout: 600
container:
  image: eval-server:v2
tasks:
  directory: /opt/tasks
logging:
  level: DEBUG
  format: json
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)

        self.assertEqual(config["server"]["port"], 9090)
        self.assertEqual(config["gpus"]["ids"], [0, 1, 2, 3])
        self.assertEqual(config["gpus"]["timeout"], 600)
        self.assertEqual(config["container"]["image"], "eval-server:v2")
        self.assertEqual(config["logging"]["format"], "json")


class FakeArgs:
    """Mock argparse namespace for testing merge logic."""

    def __init__(self, **kwargs):
        self.host = kwargs.get("host", "0.0.0.0")
        self.port = kwargs.get("port", 8080)
        self.gpus = kwargs.get("gpus", "0")
        self.timeout = kwargs.get("timeout", 530)
        self.image = kwargs.get("image", "eval-server:latest")
        self.tasks_dir = kwargs.get("tasks_dir", None)
        self.log_level = kwargs.get("log_level", "INFO")
        self.log_format = kwargs.get("log_format", "text")


class TestMergeConfigWithArgs(unittest.TestCase):
    """Test config/CLI merge logic — CLI args win over defaults."""

    def test_config_values_used_when_args_are_defaults(self):
        config = {
            "server": {"host": "127.0.0.1", "port": 9090},
            "gpus": {"ids": [0, 1], "timeout": 600},
            "container": {"image": "eval:v2"},
            "logging": {"level": "DEBUG", "format": "json"},
        }
        args = FakeArgs()
        merged = merge_config_with_args(config, args)

        self.assertEqual(merged["host"], "127.0.0.1")
        self.assertEqual(merged["port"], 9090)
        self.assertEqual(merged["gpus"], [0, 1])
        self.assertEqual(merged["timeout"], 600)
        self.assertEqual(merged["image"], "eval:v2")
        self.assertEqual(merged["log_level"], "DEBUG")
        self.assertEqual(merged["log_format"], "json")

    def test_cli_args_override_config(self):
        config = {
            "server": {"port": 9090},
            "gpus": {"ids": [0, 1]},
        }
        args = FakeArgs(port=7070, gpus="2,3")
        merged = merge_config_with_args(config, args)

        self.assertEqual(merged["port"], 7070)
        self.assertEqual(merged["gpus"], [2, 3])

    def test_empty_config(self):
        merged = merge_config_with_args({}, FakeArgs())
        self.assertEqual(merged["host"], "0.0.0.0")
        self.assertEqual(merged["port"], 8080)
        self.assertEqual(merged["gpus"], [0])
        self.assertEqual(merged["timeout"], 530)

    def test_tasks_dir_from_config(self):
        config = {"tasks": {"directory": "/opt/tasks"}}
        merged = merge_config_with_args(config, FakeArgs())
        self.assertEqual(merged["tasks_dir"], "/opt/tasks")

    def test_tasks_dir_cli_overrides_config(self):
        config = {"tasks": {"directory": "/opt/tasks"}}
        merged = merge_config_with_args(config, FakeArgs(tasks_dir="/my/tasks"))
        self.assertEqual(merged["tasks_dir"], "/my/tasks")


if __name__ == "__main__":
    unittest.main()
