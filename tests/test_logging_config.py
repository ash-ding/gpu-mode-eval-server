"""Tests for structured logging configuration."""

import json
import logging
import unittest

from eval_server.logging_config import StructuredFormatter, setup_logging


class TestStructuredFormatter(unittest.TestCase):
    """Test JSON structured log output."""

    def setUp(self):
        self.formatter = StructuredFormatter()

    def test_basic_format(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["message"], "hello world")
        self.assertEqual(data["logger"], "test")
        self.assertIn("timestamp", data)

    def test_extra_fields_included(self):
        record = logging.LogRecord(
            name="eval",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="eval done",
            args=(),
            exc_info=None,
        )
        record.gpu_id = 0
        record.task_name = "trimul"
        record.duration_ms = 1234.5
        record.success = True

        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["gpu_id"], 0)
        self.assertEqual(data["task_name"], "trimul")
        self.assertEqual(data["duration_ms"], 1234.5)
        self.assertTrue(data["success"])

    def test_extra_fields_absent_when_not_set(self):
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warn",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertNotIn("gpu_id", data)
        self.assertNotIn("task_name", data)

    def test_valid_json_output(self):
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg='special "chars" & <tags>',
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertIn("special", data["message"])

    def test_timestamp_format(self):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="ts test",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        data = json.loads(output)
        self.assertTrue(data["timestamp"].endswith("Z"))


class TestSetupLogging(unittest.TestCase):
    """Test logging setup function."""

    def tearDown(self):
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)

    def test_json_format_sets_structured_formatter(self):
        setup_logging(level="DEBUG", log_format="json")
        self.assertEqual(len(logging.root.handlers), 1)
        self.assertIsInstance(logging.root.handlers[0].formatter, StructuredFormatter)
        self.assertEqual(logging.root.level, logging.DEBUG)

    def test_text_format_sets_standard_formatter(self):
        setup_logging(level="INFO", log_format="text")
        self.assertEqual(len(logging.root.handlers), 1)
        self.assertNotIsInstance(
            logging.root.handlers[0].formatter, StructuredFormatter
        )
        self.assertEqual(logging.root.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
