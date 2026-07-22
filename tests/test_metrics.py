"""Tests for the Prometheus metrics module."""

import unittest

from eval_server.metrics import Metrics


class TestMetricsLatency(unittest.TestCase):
    """Test latency histogram observation and rendering."""

    def test_observe_latency(self):
        m = Metrics()
        m.observe_latency("trimul", 0, 5.5)
        output = m.render_prometheus()
        self.assertIn("eval_latency_seconds_sum", output)
        self.assertIn("trimul", output)
        self.assertIn("5.5", output)

    def test_latency_bucket_counting(self):
        m = Metrics()
        m.observe_latency("trimul", 0, 0.5)
        m.observe_latency("trimul", 0, 3.0)
        m.observe_latency("trimul", 0, 15.0)
        output = m.render_prometheus()
        self.assertIn('eval_latency_seconds_count{task="trimul",gpu="0"} 3', output)

    def test_cumulative_buckets(self):
        m = Metrics()
        m.observe_latency("test", 0, 0.5)
        output = m.render_prometheus()
        self.assertIn('le="+Inf"', output)
        inf_line = [l for l in output.split("\n") if '+Inf' in l and 'bucket' in l][0]
        self.assertTrue(inf_line.endswith(" 1"))


class TestMetricsQueueDepth(unittest.TestCase):
    """Test queue depth gauge."""

    def test_queue_depth(self):
        m = Metrics()
        m.set_queue_depth(8)
        output = m.render_prometheus()
        self.assertIn("queue_depth 8", output)

    def test_queue_depth_updates(self):
        m = Metrics()
        m.set_queue_depth(5)
        m.set_queue_depth(12)
        output = m.render_prometheus()
        self.assertIn("queue_depth 12", output)
        self.assertNotIn("queue_depth 5", output)


class TestMetricsEvalsTotal(unittest.TestCase):
    """Test evals counter."""

    def test_inc_success(self):
        m = Metrics()
        m.inc_evals(success=True)
        m.inc_evals(success=True)
        output = m.render_prometheus()
        self.assertIn('evals_total{success="true"} 2', output)

    def test_inc_error(self):
        m = Metrics()
        m.inc_evals(error_type="timeout")
        output = m.render_prometheus()
        self.assertIn('evals_total{error_type="timeout"} 1', output)


class TestMetricsGPUState(unittest.TestCase):
    """Test GPU worker state gauge."""

    def test_set_gpu_state(self):
        m = Metrics()
        m.set_gpu_state(0, "healthy")
        m.set_gpu_state(1, "failed")
        output = m.render_prometheus()
        self.assertIn('gpu_worker_state{gpu="0"} 1', output)
        self.assertIn('gpu_worker_state{gpu="1"} 3', output)


class TestMetricsContainerRestarts(unittest.TestCase):
    """Test container restart counter."""

    def test_inc_restarts(self):
        m = Metrics()
        m.inc_container_restarts(0)
        m.inc_container_restarts(0)
        m.inc_container_restarts(1)
        output = m.render_prometheus()
        self.assertIn('container_restarts_total{gpu="0"} 2', output)
        self.assertIn('container_restarts_total{gpu="1"} 1', output)


class TestMetricsRendering(unittest.TestCase):
    """Test overall Prometheus text rendering."""

    def test_empty_metrics(self):
        m = Metrics()
        output = m.render_prometheus()
        self.assertIn("queue_depth 0", output)

    def test_help_and_type_lines(self):
        m = Metrics()
        m.observe_latency("t", 0, 1.0)
        m.inc_evals(success=True)
        output = m.render_prometheus()
        self.assertIn("# HELP eval_latency_seconds", output)
        self.assertIn("# TYPE eval_latency_seconds histogram", output)
        self.assertIn("# HELP evals_total", output)
        self.assertIn("# TYPE evals_total counter", output)

    def test_ends_with_newline(self):
        m = Metrics()
        output = m.render_prometheus()
        self.assertTrue(output.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
