"""Multi-layer failure detection — GPU health, memory leaks, CUDA availability."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def check_gpu_health(gpu_id: int) -> tuple[bool, str]:
    """Check GPU health via nvidia-smi.

    Returns: (is_healthy, error_message)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + str(gpu_id),
                "--query-gpu=ecc.errors.corrected.volatile.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, f"nvidia-smi failed: {result.stderr.strip()}"

        parts = result.stdout.strip().split(", ")
        if len(parts) != 2:
            return False, f"Unexpected nvidia-smi output: {result.stdout.strip()}"

        ecc_errors_str, temp_str = parts

        if ecc_errors_str not in ("0", "N/A", "[N/A]"):
            try:
                ecc_count = int(ecc_errors_str)
                if ecc_count > 0:
                    return False, f"GPU has {ecc_count} ECC errors"
            except ValueError:
                pass

        try:
            temp = int(temp_str.strip())
            if temp > 85:
                return False, f"GPU temperature too high: {temp}C"
        except ValueError:
            pass

        return True, ""
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except Exception as e:
        return False, f"Health check failed: {e}"


def check_memory_leak(gpu_id: int, baseline_mb: int) -> tuple[bool, int]:
    """Check if GPU memory usage has grown beyond baseline + 2GB.

    Returns: (has_leak, current_usage_mb)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + str(gpu_id),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, 0

        current_mb = int(result.stdout.strip())
        has_leak = current_mb > baseline_mb + 2048
        return has_leak, current_mb
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False, 0
