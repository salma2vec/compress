# The output head explains most of the rowwise-NF4 failure

Measured on 2026-10-14 with the existing Pythia-70M/WikiText-2 pilot assets on
Apple M1 MPS, CPU fallback disabled. The [protocol](failure-diagnosis-protocol.md)
and runner were committed at `eda071efb5aa39b303b6f0119072d5ea6e48e5ea` before
execution. This is an exploratory intervention study using the same previously
observed 16 paragraphs (3,090 predicted tokens), not fresh confirmation.

| Intervention | Perplexity | Persistent tensor bytes |
| --- | ---: | ---: |
| FP32 | 79.408983 | 281,706,560 |
| All linear modules INT8 | 80.971919 | 148,128,320 |
| All linear modules rowwise NF4 | 2,139,132.272676 | 125,813,312 |
| NF4 body, INT8 output head | 107.941737 | 138,691,136 |
| FP32 body, NF4 output head | 1,399,282.034628 | 191,763,008 |

The all-NF4 excess mean NLL over FP32 was 10.201299 nat/token. Keeping only the
output head at INT8 removed **96.9908%** of this excess. Quantizing only the head
to NF4 reproduced **95.8393%**. Both exceed the frozen 80% dominance thresholds.
These intervention effects need not add to one: the head and body interact.

Every method passed identical save/reload tensor hashes, exact CPU logits after
reload, independently shifted-token loss, complete paragraph token counting and
first-window CPU/MPS logit parity (atol=rtol=1e-4). Scalar-oracle unit tests also
checked the quantizers on signed, zero, constant, outlier and odd-sized matrices.
The [raw report](../benchmarks/results/pythia70m-nf4-diagnosis.json) retains all
method results, module residuals, paragraph losses, hashes and provenance.

This is a concrete failure diagnosis, not a successful low-bit-quality result:
the protected NF4 body still has perplexity 107.94 versus FP32's 79.41. The
interventions have different storage, so this comparison is not evidence of
matched-budget superiority. It evaluates this rowwise codebook implementation
on one model and observed sample, not QLoRA or NF4 generally. No throughput,
peak-memory, CUDA or model-generalization conclusion follows.

The supported policy now floors declared output heads at INT8, permits an
explicit opt-out, and records policy in precision plans and packed checkpoints.
FP16 heads retain FP16. Simulation also protects an exactly tied embedding/head
weight when the embedding alias is visited first; packed tied weights remain
unsupported. Existing archives are restored exactly rather than re-quantized
under the new default. A new calibration-only allocation study with unseen
article groups is required before claiming a useful compression improvement.
