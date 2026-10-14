# compress

A reproducible workbench for activation-guided, weight-only quantization. Profile a calibration distribution, assign precision to weighted modules, store packed weights, and measure the held-out quality/storage tradeoff against uniform and shuffled controls.

The central question is empirical: **does activation magnitude help allocate a fixed precision budget?** The library implements a simple magnitude × density heuristic. It does not assume that this heuristic improves quality or that a small stored model runs faster.

## Install and run

```bash
git clone https://github.com/salma2vec/compress.git
cd compress
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
python -m benchmarks.offline --seed 42 --steps 100 --output results/offline.json
```

Python 3.11+ and PyTorch 2.6+ are required. The offline benchmark needs no downloaded model, dataset, API key, or accelerator. It trains a small causal attention model on a synthetic cyclic-symbol task, uses separate calibration and evaluation sequences, and writes configuration, corpus hashes, raw timing samples, likelihoods, checkpoint sizes and tensor storage bytes. This is a functional integration experiment, not a pretrained-language-model benchmark.

A measured [CPU reference run](docs/reference-run.md) is included with its raw report.

A [pretrained Pythia-70M / WikiText-2 experiment](docs/pretrained-validation.md) adds real held-out text, exact-storage shuffled controls, saved/reloaded checkpoints and Apple MPS evaluation. It retains a severe uniform-NF4 failure and explicitly distinguishes its small paragraph sample from the standard full-test benchmark.

## Two quantization paths

| Path | Representation | Purpose |
| --- | --- | --- |
| `ActivationAwareQuantizer` | Original floating-point parameters with quantization error simulated | Broad compatibility and quality ablations; no storage compression |
| `pack_model` | FP16 buffers, symmetric INT8, or two NF4 indices per byte | Actual weight storage reduction for ordinary `nn.Linear` modules |

The packed backend dequantizes weights before each PyTorch matrix multiplication. Persistent storage can shrink while latency and peak working memory increase. It has no fused low-bit inference kernel. NF4 uses the QLoRA codebook with per-output-row scaling, not QLoRA's full blockwise quantization/finetuning pipeline.

```python
from compress import ActivationRecorder, ImportanceScorer, MixedPrecisionConfig, QuantizationConfig
from compress.compression.packed import pack_model, save_packed, load_packed, tensor_bytes

# model, tokenizer, calibration_texts are supplied by the caller.
profile = ActivationRecorder(model).record(calibration_texts, tokenizer, n_batches=100)
scores = ImportanceScorer(profile).layer_importance()
plan = MixedPrecisionConfig.from_importance(scores)
plan.to_json('precision.json')
model = pack_model(model, QuantizationConfig(plan.layer_dtypes))
print(tensor_bytes(model))
save_packed(model, 'model.safetensors')
# Rebuild the same architecture explicitly; no Python model objects are unpickled.
restored = load_packed(build_model(), 'model.safetensors')
```

Packed conversion is inference-only and modifies the model in place. It rejects tied linear weights and `nn.MultiheadAttention`, which reads child weights directly. Models using GPT-2-style `Conv1D`, custom weight access, shared module aliases, or other fused projection APIs require an adapter; use the simulation path for those architectures. Validate outputs before deploying a new model family.

## Evaluation contracts

- Calibration failures are visible, hooks are cleaned up on failure, and tensors move to the model device. Statistics are weighted by activation element count.
- Importance is normalized per profile; it is a module-level heuristic, not a causal sensitivity estimate or an individual-head score.
- Perplexity evaluates complete documents using overlapping windows and counts each predicted target once. Model failures and nonfinite losses stop evaluation.
- Generation latency synchronizes CUDA/MPS and counts tokens actually emitted, including early stopping. Raw samples accompany the summary.
- The offline comparison includes FP32, uniform INT8, uniform NF4, mixed precision and a mixed assignment shuffled within equal-shaped layer groups. The latter preserves the same stored-byte budget.
- Checkpoints use safetensors. Hugging Face CLI model loading also requires safetensors and does not enable remote model code.

## Hugging Face simulation workflow

```bash
compress profile --model gpt2 --data calibration.txt --out profile.json
compress quantize --model gpt2 --profile-path profile.json --out simulated-model
compress evaluate --model simulated-model --data heldout.txt
compress benchmark --model gpt2
```

The `quantize` CLI saves fake-quantized floating-point weights in a normal Hugging Face directory. It does **not** produce packed low-bit weights; use `pack_model` for supported custom architectures. The downloaded-model benchmark uses disjoint calibration/evaluation samples but remains a synthetic smoke experiment unless you supply a representative corpus. Do not use its latency to infer low-bit kernel acceleration.

## Research scope

`profiling/`, `compression/`, `evaluation/`, and `reconstruction/` separate measurement from interventions. Auxiliary APIs include attention **weight ablation** (`HeadPruner`, a legacy name), layer selection, weight transplantation and bias correction. These are experimental utilities; weight ablation does not remove heads or reduce parameter count, and layer selection does not train a distilled student.

See [methodology and related work](docs/methodology.md) for assumptions, limitations and current research context. No SOTA result or same-budget quality improvement is claimed. A publication-scale study still requires representative pretrained models, multiple seeds/corpora, matched budgets, hardware-aware low-bit kernels and external baselines.
