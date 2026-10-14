# Portable INT8 embedding result

Measured 2026-10-14 on CPU from frozen source `6424abe`, using the
[prospective protocol](embedding-int8-validation-protocol.md). The deterministic
seed-1043 panel contains two paragraphs from each of eight WikiText-2 articles,
3,098 predicted tokens total. It excludes every complete article exposed by the
earlier pilot and allocation panel, plus exact calibration-text duplicates. This
is a newly sampled local panel, not the standard concatenated WikiText test set.

| Saved method | Persistent tensor bytes | Reduction vs FP32 | Perplexity |
| --- | ---: | ---: | ---: |
| FP32 | 281,706,560 | — | 88.088675 |
| Rowwise INT8 linear modules | 148,128,320 | 47.42% | 89.268630 |
| Rowwise INT8 linear modules + input embedding | 71,062,592 | 74.77% | 89.373026 |

The embedding candidate's perplexity ratio to FP32 was **1.014580**, below the
frozen maximum of 1.05, and its byte count exactly matched the prospective
71,062,592-byte target. The practical gate therefore passed. Compared with the
linear-only INT8 checkpoint, packing the 50,304 by 512 input embedding removes a
further 77,065,728 persistent bytes while changing sampled perplexity by 0.117%.

All three methods scored the same 3,098 targets. Each saved and reloaded through
safetensors with identical state-tensor hashes and exact CPU check-window logits;
independent shifted loss, paragraph target counts and per-paragraph arithmetic
also passed. Linear-only archives retain format version 1. Archives containing a
packed embedding use version 2 and validate embedding names, precision, padding
semantics, tensor schema, finite scales and architecture compatibility before
changing the caller's model.

The compact public [result summary](../benchmarks/results/pythia70m-embedding-int8-cpu-summary.json)
records the frozen hashes, panel rows, metrics and correctness gates. The complete
private raw report has SHA-256
`561eaf425472d14459efdcbbe29deb19d4945d6a08a568b9a34b69245c2b739c`;
an independent standard-library replay produced SHA-256
`4e091826bcf83f3eb2aa6c4f2a903198f2bd5b4f3a0835b4bb7ef5d764b52169`.

This result supports portable stored-size reduction and sampled language-model
quality for one small model and corpus panel. The backend reconstructs floating
rows and weights for ordinary PyTorch operations. No inference-speed, peak-memory,
energy, multi-model or state-of-the-art claim follows. AWQ, CASA and MixQuant are
discussed in the protocol as related primary work; this experiment reproduces
none of their methods or reported results.
