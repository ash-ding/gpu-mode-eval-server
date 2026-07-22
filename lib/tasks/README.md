# GPU Mode Task Definitions

This directory contains task definitions for GPU kernel evaluation benchmarks.

## Available Tasks

### 1. Trimul (Triangle Multiplicative Update)

**Directory:** `trimul/`

**Description:** Implement the Triangle Multiplicative Update (TriMul) operator from AlphaFold3.  
A core operation for protein structure prediction models in BioML.

**Files:**
- `task.yml` — Task specification (18 test cases, 7 benchmark configs)
- `task.py` — Input/output schema
- `reference.py` — PyTorch reference implementation
- `eval.py` — Evaluation harness (correctness tests + performance benchmarks)
- `utils.py` — Helper utilities
- `submission.py` — Example submission template

**Test cases:** 18 correctness tests (various sequence lengths, batch sizes, distributions)  
**Benchmarks:** 7 performance benchmarks (geometric mean of runtimes)

**Target hardware:** H100, A100, B200, MI300X

---

### 2. MLA Decode (Multi-Head Latent Attention Decode)

**Directory:** `mla_decode/`

**Description:** Implement the MLA Decode operation for efficient transformer inference.

**Files:**
- `task.yml` — Task specification
- `task.py` — Input/output schema
- `reference.py` — PyTorch reference implementation
- `eval.py` — Evaluation harness
- `utils.py` — Helper utilities
- `submission.py` — Example submission template

---

## Task Structure

Each task follows this standard structure:

```
<task_name>/
├── task.yml          # Task specification (tests, benchmarks, timeouts)
├── task.py           # Input/output type definitions
├── reference.py      # Reference implementation (correctness baseline)
├── eval.py           # Evaluation harness (runs tests + benchmarks)
├── utils.py          # Helper functions
└── submission.py     # Example submission template
```

## Evaluation Flow

When a kernel is submitted to the eval server:

1. **Compile:** Load user kernel code with `load_inline()` or import
2. **Correctness Tests:** Run all test cases from `task.yml`
   - Compare output against reference implementation
   - Check numerical accuracy (default tolerance: 1e-4)
3. **Performance Benchmarks:** If all tests pass, run benchmarks
   - Measure execution time across multiple configurations
   - Compute geometric mean → `score_us` (microseconds)
4. **Return Result:** Comprehensive response with logs, timing, test details

## Adding New Tasks

To add a new task:

1. Create directory: `lib/tasks/<task_name>/`
2. Add required files (task.yml, task.py, reference.py, eval.py)
3. Define test cases and benchmarks in `task.yml`
4. Implement reference solution in `reference.py`
5. Update this README

## Sources

These task definitions were extracted from:
- [test-time-training/discover](https://github.com/test-time-training/discover)
- [GPU Mode reference-kernels](https://github.com/gpu-mode/reference-kernels)

Trimul competition details: https://tinyurl.com/gpumode-trimul
