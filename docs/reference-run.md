# CPU reference run

Run on 2026-10-14 with Python 3.11.15, PyTorch 2.14.1, seed 42, one CPU thread and 100 training steps. [Raw report](../benchmarks/results/offline-cpu-seed42.json) includes split hashes and timing samples.

| Method | Held-out perplexity | Persistent tensor bytes |
| --- | ---: | ---: |
| fp32 | 1.017227 | 29760 |
| uniform_int8 | 1.017253 | 10624 |
| uniform_int4 | 1.018071 | 7296 |
| mixed | 1.017233 | 11968 |
| mixed_shuffled | 1.017233 | 11968 |

This is the synthetic cyclic-symbol experiment described in the README. Packed models were saved and reloaded before evaluation. The tiny mixed-versus-shuffled difference is not evidence of a repeatable quality improvement. No pretrained-language-model result or GPU speedup is established. Persistent tensor bytes exclude temporary dequantization allocations; low-bit stored weights still use ordinary floating-point matrix multiplication.
