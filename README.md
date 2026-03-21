# compress

**Activation-aware compression for transformer models.**

Standard quantization treats all weights equally. `compress` profiles which attention heads and MLP neurons are most active on *your specific data distribution*, then applies more aggressive compression to the dormant ones. The result: better task-specific quality at the same compression ratio, or higher compression at the same quality.

---

## Why activation-aware?

A transformer trained on general text will use different layers heavily depending on what it's actually doing. A model deployed for code generation activates different heads than one running document summarization. Uniform INT8 quantization ignores this entirely.

`compress` profiles the model on a calibration corpus representative of your deployment distribution, computes a per-layer importance score, and assigns quantization precision accordingly:

- **High importance** (active, dense) → FP16
- **Medium importance** → INT8
- **Low importance** (sparse, near-zero activations) → INT4 (NF4)

In practice this means critical layers stay sharp while dormant layers are aggressively compressed, improving the quality/size tradeoff over uniform schemes.

---

## Installation

```bash
pip install compress
# or for development:
git clone https://github.com/salma2vec/compress.git
cd compress
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.0, Transformers ≥ 4.35

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from compress import (
    ActivationRecorder,
    ImportanceScorer,
    ActivationAwareQuantizer,
    QuantizationConfig,
    MixedPrecisionConfig,
)

# Load your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 1. Profile activations on calibration data
recorder = ActivationRecorder(model)
texts = ["Your calibration text here.", "More representative samples..."]
profile = recorder.record(texts, tokenizer, n_batches=100)

# 2. Compute per-layer importance scores
scorer = ImportanceScorer(profile)
importance = scorer.layer_importance()
print(scorer.summary())
# {'n_layers': 144, 'mean_importance': 0.41, 'dtype_breakdown': {'fp16': 18, 'int8': 52, 'int4': 74}}

# 3. Build a mixed-precision config from importance scores
mp_config = MixedPrecisionConfig.from_importance(importance)
mp_config.to_json("my_model_config.json")  # Save for reproducibility

# 4. Quantize
q_config = QuantizationConfig(layer_dtypes=mp_config.layer_dtypes)
quantizer = ActivationAwareQuantizer(q_config)
model = quantizer.quantize_model(model)
print(quantizer.compression_report())
# {'fp16': 18, 'int8': 52, 'int4': 74}

# 5. Measure the result
from compress import compute_perplexity, measure_latency

ppl = compute_perplexity(model, tokenizer, texts[:50])
lat = measure_latency(model, tokenizer, n_tokens=50)
print(f"Perplexity: {ppl:.2f}")
print(f"Throughput: {lat['tokens_per_sec']:.1f} tok/s")
```

### Using the CLI

```bash
# Profile a model
compress profile --model gpt2 --out profile.json --n-batches 100

# Quantize using the profile
compress quantize --model gpt2 --profile-path profile.json --out ./gpt2-compressed/

# Evaluate
compress evaluate --model ./gpt2-compressed/

# Run the full benchmark
compress benchmark
```

---

## Architecture

```
compress/
├── profiling/
│   ├── activation_recorder.py   # Forward hook-based activation profiler
│   ├── importance.py            # Per-layer importance scoring
│   └── calibration_data.py     # Calibration corpus utilities
├── compression/
│   ├── quantization.py          # INT8, NF4, activation-aware quantizer
│   ├── pruning.py               # Structured head pruning
│   ├── mixed_precision.py       # Layer → dtype assignment
│   └── distillation.py         # Layer-drop distillation
├── reconstruction/
│   ├── correction.py            # Post-compression bias correction
│   └── weight_surgery.py       # Weight transplantation utilities
├── evaluation/
│   ├── perplexity.py            # Language model perplexity
│   ├── latency.py               # Token generation latency
│   └── task_eval.py            # Classification and MC accuracy
└── cli.py                       # Typer CLI
```

**Pipeline:** `ActivationRecorder` → `ImportanceScorer` → `MixedPrecisionConfig` → `ActivationAwareQuantizer` → evaluation

---

## Benchmark

Run `make benchmark` or `compress benchmark` to reproduce.

| Method                       | Perplexity | Tokens/sec | PPL delta |
|------------------------------|------------|------------|-----------|
| Baseline (FP32)              | —          | —          | —         |
| Uniform INT8                 | —          | —          | —         |
| Activation-aware mixed       | —          | —          | —         |
| Mixed + head pruning (30%)   | —          | —          | —         |

*Fill in after running `make benchmark` on your hardware.*

---

## API Reference

### `ActivationRecorder`

```python
class ActivationRecorder(model: nn.Module, epsilon: float = 1e-3)
```

Attaches forward hooks to all attention and MLP submodules and accumulates per-layer activation statistics across a calibration corpus.

| Method | Description |
|--------|-------------|
| `attach()` | Register forward hooks on all `attn`/`mlp`/`attention`/`feed_forward` submodules |
| `detach()` | Remove all registered hooks |
| `reset()` | Clear accumulated statistics |
| `record(inputs, tokenizer, n_batches)` | Run forward passes and return `ActivationProfile` |

**`ActivationProfile`** exposes `layers: dict[str, LayerStats]`. Each `LayerStats` has `mean_magnitude`, `sparsity`, `variance`, `n_samples`.

---

### `ImportanceScorer`

```python
class ImportanceScorer(profile: ActivationProfile)
```

Computes normalized per-layer importance scores from an activation profile.

| Method | Description |
|--------|-------------|
| `layer_importance()` | Returns `dict[str, float]` in [0, 1] — composite of magnitude × density |
| `head_importance(model, calibration_inputs)` | Attention-only importance scores |
| `recommended_dtype(layer_name)` | Returns `"fp16"`, `"int8"`, or `"int4"` |
| `summary()` | Returns counts and statistics |

---

### `ActivationAwareQuantizer`

```python
class ActivationAwareQuantizer(config: QuantizationConfig)
```

Applies mixed-precision quantization to a model in-place.

| Method | Description |
|--------|-------------|
| `quantize_model(model)` | Apply per-layer dtype assignments from config, return model |
| `compression_report()` | Returns `dict[str, int]` with counts per dtype |

Supported dtypes:
- `"fp16"` — cast to half and back (precision reduction, no bit-packing)
- `"int8"` — per-channel symmetric INT8 with scale/zero-point
- `"int4"` — 4-bit NormalFloat (QLoRA-style NF4)

---

### `HeadPruner`

```python
class HeadPruner(keep_fraction: float = 0.7)
```

Structured pruning of attention layers ranked by importance.

| Method | Description |
|--------|-------------|
| `prune_heads(model, importance)` | Zero out bottom `(1 - keep_fraction)` of attention layers |
| `pruning_report()` | Returns `{"n_pruned": int, "pruned_layers": list[str]}` |

---

## Adding a New Compression Method

1. Create `compress/compression/my_method.py`
2. Implement a class with a `compress_model(model, importance) -> nn.Module` interface
3. Register it in `compress/compression/__init__.py`
4. Add tests in `tests/test_my_method.py`
5. Wire it into the benchmark in `benchmarks/gpt2_compression.py`

The `importance: dict[str, float]` dict from `ImportanceScorer.layer_importance()` is the canonical interface for passing activation-guided signals to compression methods.

---

## License

Apache 2.0
