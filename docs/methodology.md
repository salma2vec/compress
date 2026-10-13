# Methodology and related work

Updated 2026-10-14. This project tests an intentionally small hypothesis: the product of mean absolute module output and activation density may help choose module precision. Thresholds map normalized scores to FP16, INT8 and NF4. Thresholding does not optimize a global byte budget. The equal-shape shuffled control tests whether the allocation signal helps at the exact same stored-byte budget.

Calibration, training and evaluation must be disjoint. Distribution shift, layer rescaling, activation outliers and module-size differences can undermine magnitude-based ranking. Report these failure cases instead of selecting only successful runs. Stored tensor bytes include scales and biases; on-disk bytes additionally include metadata. Neither metric is peak runtime memory. The portable backend allocates dequantized weights during inference.

## Relation to primary work

- [AWQ (Lin et al., 2023/MLSys 2024)](https://arxiv.org/abs/2306.00978) uses activation information to protect salient weights through scaling. Our module-level precision heuristic is not an AWQ implementation or reproduction.
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) introduces NF4 within a memory-efficient finetuning system. We reuse its codebook, with rowwise scaling and packed storage; double quantization, paged optimizers and LoRA training are not implemented.
- [FAQ (Lv et al., 2026)](https://arxiv.org/abs/2602.02538) considers future-layer activations and calibration robustness. This highlights a limitation of our local activation statistic; future-layer scoring is not implemented.
- [TWLA (2026)](https://arxiv.org/abs/2606.13054) studies ternary weights and low-bit activations. The current project is weight-only, and does not implement its activation quantization or reproduce its results.
- [Hugging Face perplexity guidance](https://huggingface.co/docs/transformers/perplexity) motivates overlapping windows and accounting for causal label shifts. Our evaluator reports full-document token-weighted likelihood, preserving document boundaries.

## Reproduction

Run `python -m benchmarks.offline --seed 42 --steps 100 --output results/offline.json` after installing the project. Keep the JSON alongside the exact source revision and package versions. The output records model-training steps, data hashes, precision assignments, raw timing samples and held-out metrics. The synthetic cyclic-symbol task only checks the end-to-end implementation. It cannot establish practical language-model quality, compression superiority or deployment speed.

Benchmark multiple seeds before claiming an effect. For pretrained models, split representative text before calibration, match actual packed byte counts, compare an independently implemented uniform baseline, evaluate several task families, and report failures. Use dedicated supported low-bit kernels if latency is the objective.
