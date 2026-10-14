# Calibration-only allocation: prospective local protocol

Version 1, 2026-10-14. Freeze source and a separate paragraph manifest before
running model scores. This follows the observed NF4 output-head diagnosis; the
new test paragraphs exclude every article containing an old pilot paragraph.
No result is claimed by this protocol. It is a locally prospective specification,
not external preregistration. No downloads, paid calls or model training.

Use only the previously cached pinned Pythia-70M safetensors, tokenizer and
WikiText-2 train/test Parquet assets, verified against the recorded pilot hashes.
For each calibration seed 42, 142, 242, 342, 442, sample eight eligible train
paragraphs and use the first 128 tokens. Both scores see exactly these tokens:

- Existing activation magnitude × density, recorded on CPU.
- Direct isolated intervention: mean calibration NLL with only module j NF4,
  minus mean NLL with only j INT8; all other weights remain FP32. Compute shifted
  loss directly from logits. Divide by the additional persistent tensor bytes
  needed for INT8 to report benefit per byte. Negative scores are retained.

Packing is on CPU for both calibration interventions and final candidates.
Inference may use explicitly available MPS with CPU fallback disabled. Restore
the original module after every intervention, including exceptions. Keep the
declared `lm_head` at INT8; it is not eligible for allocation or rescored.
The direct score is an established perturbation technique. It neglects joint
interactions; no novelty or universal optimality is claimed.

Within each identical weight/bias-shape group, promote the higher-scoring half
(rounded up) to INT8 and keep the rest NF4. Singletons are INT8. This constrained
allocation matches the old magnitude plan and three within-shape shuffled
controls (seeds 11, 23, 37) exactly, including scales and biases. It is not an
unrestricted global byte-budget optimizer. Since promotion cost is equal within
each group, dividing by bytes leaves that group's ordering unchanged.

Before any held-out loss, freeze all five seeds' scores and assignments. Limit
scoring to the 24 non-head linear modules × two precisions × eight paragraphs ×
five seeds = 1,920 forward calls. No score tuning or fallback allocation based on
test results. Retain each calibration seed even when the score is unfavorable.

Assign article groups from top-level WikiText headings before filtering tokens.
Exclude whole articles containing any of the 16 previously observed paragraphs,
and exact text duplicates of any current or prior-pilot calibration paragraph. Eligible paragraphs are
128–512 tokens. With fixed seed 1042, sample 16 eligible articles with at least
two paragraphs, then two paragraphs per article: 32 total. This article-balanced
sample is not the standard concatenated full-test WikiText benchmark. If there
are too few eligible articles, stop before model losses; do not silently relax
the criteria. Freeze indices, article IDs and text hashes in a new exclusive file.

Score all held-out tokens after each paragraph's first token once, using context
256 and stride 128. Evaluate FP32, all-INT8 and protected-head NF4 references,
plus all five matched-budget policies for each calibration seed. Independently
verify exact tensor hashes and CPU logits across save/reload, first-window
CPU/accelerator logits (atol=rtol=1e-4), shifted-token loss and target counts.
Retain paragraph/article IDs and loss sums so paired article-level deltas can be
recomputed. Remove temporary checkpoints only after their hashes are recorded.
Do not report throughput or peak-memory claims from this study.

The **score gate** requires exact mixed-policy tensor-byte equality, all checks
passing, at least 0.01 nat/token improvement over magnitude and improvement over
the median shuffled policy in at least four of five seeds, with no seed more
than 0.02 nat/token worse than magnitude. The separate **practical-quality gate**
requires perplexity at most 1.05 × FP32 at no more than 143,409,728 persistent
tensor bytes in at least four of five seeds. All denominators and failures must
be reported. Neither gate is a significance test; all seeds share one held-out
article sample. One model cannot establish model-generalization or superiority
over contemporary compression methods. Held-out means excluded from calibration
and the old pilot, not known absent from Pythia pretraining.

Related primary work: [AWQ](https://arxiv.org/abs/2306.00978v6) studies activation-
aware low-bit quantization; [CASA](https://arxiv.org/abs/2609.25916) addresses
limitations of scalar sensitivity and joint allocation; [random-probe mixed
precision allocation](https://arxiv.org/abs/2609.33923v2) also studies sensitivity
under exact storage budgets. This modest direct-intervention baseline reproduces
none of those methods and claims no state-of-the-art result.
