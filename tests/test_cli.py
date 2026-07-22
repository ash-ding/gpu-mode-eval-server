"""Tests for the CLI evaluation script."""

import os
import tempfile
import unittest

from eval_server.cli import main


class TestCLIParsing(unittest.TestCase):
    """Test CLI argument handling."""

    def test_missing_file_exits(self):
        import sys

        old_argv = sys.argv
        sys.argv = [
            "eval-kernel",
            "--server",
            "http://localhost:8080",
            "--code",
            "/nonexistent/kernel.py",
            "--task",
            "trimul",
        ]
        try:
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 1)
        finally:
            sys.argv = old_argv


class TestCLIModule(unittest.TestCase):
    """Test that cli module is importable and main is callable."""

    def test_import(self):
        from eval_server import cli

        self.assertTrue(callable(cli.main))


if __name__ == "__main__":
    unittest.main()
