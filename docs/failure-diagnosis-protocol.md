# Prospective failure diagnosis: Pythia-70M rowwise NF4

Protocol version 1, 2026-10-14. Freeze this protocol and runner in a clean commit
before execution. No measurements from this protocol are claimed here.

The previously observed uniform-NF4 failure is the target. This is exploratory
diagnosis using the same 16 WikiText-2 pilot paragraphs, not a new held-out claim.
Assets and paragraph hashes must match the retained pilot. Use only existing
local Pythia-70M safetensors and cached test Parquet; downloads and remote model
code are disabled. Evaluate FP32 with eager attention, disabled caches, one CPU
thread and seed 42. MPS is optional, explicitly selected, with CPU fallback off.

Run all five predetermined interventions without changing quantizer arithmetic:

1. Original FP32.
2. All 25 ordinary linear modules INT8.
3. All 25 linear modules rowwise NF4.
4. Rowwise-NF4 body with only `lm_head` INT8.
5. FP32 body with only `lm_head` rowwise NF4.

An omitted module in a partial intervention must remain FP32, not inherit a
default compressed dtype. These methods intentionally differ in stored bytes;
their comparison diagnoses mechanism, not matched-budget superiority. Preserve
per-module weight residuals and actual storage. Save/reload each intervention,
require identical tensor hashes and CPU logits, then require CPU/device first-
128-token-window parity at atol=rtol=1e-4. Fail visibly on any infrastructure,
roundtrip, nonfinite, parity or token-coverage problem; retain an incomplete
report. Stream one candidate/checkpoint at a time.

Evaluate every paragraph using context 256 / stride 128 and score every target after
the first token exactly once. Report per-paragraph NLL/token counts and aggregate
token-weighted NLL. No retries, calibration, learned scoring or tuning occurs.

Let `L` denote mean NLL and `D=L(all-NF4)-L(FP32)`. If `D<=1` nat/token, label the
prior severe failure as not reproduced and do not evaluate a dominance claim.
Otherwise the descriptive **dominant head failure** hypothesis is supported only
when both fractions are at least 0.80:

```
rescue   = [L(all-NF4) - L(NF4-body, INT8-head)] / D
isolated = [L(FP32-body, NF4-head) - L(FP32)] / D
```

Report rejection if either fails. Fractions can lie outside [0,1] and need not
sum to one because modules interact. A rejected hypothesis is a completed
scientific outcome; an infrastructure failure is not. Do not infer NF4/QLoRA
quality in other implementations, general compression superiority, latency,
peak-memory savings, or causal additivity from these interventions.

Independent tests compare packed INT8/NF4 values to a scalar nearest-codebook
reference on signed, zero, constant, odd-sized and outlier matrices. Later
allocation changes require a new protocol and unseen article-grouped evaluation.
