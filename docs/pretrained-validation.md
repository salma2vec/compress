# Pythia-70M on sampled WikiText-2

Measured on 2026-10-14 using an Apple M1 (7 GPU cores, 8 GiB unified memory),
macOS 14.6, Python 3.11.15, PyTorch 2.14.1 and Transformers 5.19.0. Forward
evaluation used actual MPS with CPU fallback disabled. Calibration and packing
used CPU. There was no paid inference or cloud compute.

The pretrained model is [EleutherAI/Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m),
revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`, loaded from safetensors
without remote model code. Text comes from
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, subset `wikitext-2-raw-v1`.
The public sources document the model's Apache-2.0 and corpus's CC-BY-SA/GFDL terms;
the repository retains measurements and hashes, not redistributed corpus text or
model weights.

## Protocol

Calibration selected 32 distinct train-split paragraphs with seed 42 and used
their first 128 tokens. Evaluation selected 16 distinct test-split paragraphs
with seed 43, totaling **3,090 predicted tokens**. Eligible paragraphs contain
128–512 tokens; blank lines, headings and exact duplicate paragraphs are excluded.
Paragraph selection is made without inspecting model loss. Train/test selections
were checked for exact text overlap. Each complete evaluation paragraph was scored
with a 256-token window and 128-token stride, excluding its first token.

This sampled, paragraph-separated evaluation is **not the standard concatenated
full-test WikiText perplexity benchmark**. It also does not establish that the
corpus was absent from Pythia's original pretraining data.

The activation policy ranks mean output magnitude × activation density within
identical linear weight/bias shapes. The highest half receive INT8 and the rest
row-scaled NF4; singleton groups receive INT8. This fixed policy was specified
before evaluation, without tuning on held-out losses. Three seeded shuffles
permute precision within the same groups, preserving exactly the same persistent
tensor storage. Uniform controls use the same quantizers, not an independent
AWQ/GPTQ/QLoRA implementation.

Every method, including FP32, was saved and reloaded from safetensors before
evaluation. Every restored tensor's byte hash matched. An independent explicit
shifted-token cross-entropy check agreed with the model's reported loss. Eager
attention and FP32 arithmetic were used; caches were disabled. The portable packed
backend dequantizes before ordinary matrix multiplication.

## Measurements

| Method | Held-out perplexity | Persistent tensor bytes |
| --- | ---: | ---: |
| FP32 | 79.408983 | 281706560 |
| Uniform INT8 | 80.971919 | 148128320 |
| Uniform NF4 | 2139132.272676 | 125813312 |
| Activation-ranked mixed | 86.448360 | 143409728 |
| Equal-storage shuffle, seed 11 | 87.942329 | 143409728 |
| Equal-storage shuffle, seed 23 | 94.560182 | 143409728 |
| Equal-storage shuffle, seed 37 | 91.607146 | 143409728 |

Uniform NF4 failed badly on this model. The result is retained. A separate
CPU-versus-MPS check on the first held-out 128-token window reproduced its poor
loss on CPU (15.539417 versus MPS 15.539341). FP32, INT8 and NF4 logits all passed
the recorded relative/absolute tolerance (`rtol=atol=1e-4`) on that window. This
supports a quantization failure rather than a gross MPS-only implementation error;
one-window parity is not an exhaustive accelerator proof.

The ranked allocation outperformed the three equal-storage shuffles in this
sample. One model, one calibration subset and 16 evaluation paragraphs cannot
establish a reliable general advantage. Paragraphs may share an article, and
shuffle variation is not a confidence interval over models or corpora. The large
single output projection stays INT8 in the mixed/shuffled policies; this is a
substantial difference from uniform NF4. No conclusion about NF4 in other
implementations follows from the failure here.

Seven synchronized 128-token prefill timings per method are retained, after three
warmups and in a seeded method order. This shared workstation was under memory
pressure; those timings are diagnostic, not evidence for low-bit acceleration.
No CUDA, optimized low-bit kernel, peak-memory benefit or training result is claimed.

## Evidence and reproduction

- [Raw model results](../benchmarks/results/pythia70m-wikitext2-mps-seed42.json): per-paragraph NLL/token counts, checkpoint hashes/sizes, assignments, activation statistics and timing samples.
- [CPU/MPS comparison](../benchmarks/results/pythia70m-wikitext2-device-parity.json).
- [Provenance](../benchmarks/results/pythia70m-wikitext2-provenance.json): exact executed harness revision/hash, report hash and package versions. Metadata added after the run is identified explicitly.

```sh
pip install -e '.[dev]'
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m benchmarks.pretrained --device mps --seed 42 \
  --calibration-count 32 --evaluation-count 16 --output results/pythia-wikitext
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m benchmarks.device_parity \
  --report results/pythia-wikitext/report.json \
  --output results/pythia-wikitext/device-parity.json --device mps
```

Pin package versions from the provenance file for a closer reproduction. Use
`--device cpu` for CPU evaluation, or `--device cuda` on CUDA hardware; each is a
distinct experiment. The first command downloads only the pinned public assets.
The diagnostic reuses those local assets. Existing outputs are rejected. Temporary
checkpoints are now deleted after each method's verified metrics are saved, bounding
disk use rather than retaining all seven checkpoints together.
