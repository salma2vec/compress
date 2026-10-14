# Portable INT8 embedding: synthetic second-domain check

Measured 2026-10-14 on CPU from frozen source `74b45e6`, using the
[prospective protocol](embedding-int8-second-corpus-protocol.md). The panel is
the complete 50-record test split from a local MIT-licensed synthetic incident
generator: 10,568 predicted tokens in total. All input documents, token counts,
source hashes, thresholds, and method order were frozen before model loading.

This corpus is **small, synthetic, and template-generated**. The result is a
second-domain engineering check for checkpoint storage and gross quality
regression. It is not evidence of performance on a second natural corpus or of
generalization to another model.

| Saved method | Persistent tensor bytes | Reduction vs FP32 | Perplexity |
| --- | ---: | ---: | ---: |
| FP32 | 281,706,560 | — | 93.591605 |
| Rowwise INT8 linear modules | 148,128,320 | 47.42% | 96.463741 |
| Rowwise INT8 linear modules + input embedding | 71,062,592 | 74.77% | 96.249124 |

The embedding candidate's perplexity ratio to FP32 was **1.0283948383**, below
the prospectively frozen maximum of 1.05. Its persistent tensor count exactly
matched the 71,062,592-byte target, so the practical gate passed. Packing the
input embedding removed another 77,065,728 bytes, or 52.03%, from the
linear-only INT8 checkpoint. Its slightly lower perplexity than linear-only
INT8 on this panel is descriptive and does not support a superiority claim.

All three methods scored the same 50 documents and 10,568 targets. Each method
saved and reloaded through safetensors with identical state-tensor hashes and
exact CPU check-window logits. Independent shifted-loss checks passed. A
separate standard-library replay verified document identity, target counts,
all 150 per-document NLL records, aggregate mean NLL and perplexity, checkpoint
correctness flags, exact byte counts, the acceptance ratio, and the previously
published tensor-state hashes.

The recomputable public [result record](../benchmarks/results/pythia70m-embedding-int8-second-corpus-cpu-summary.json)
contains the frozen document hashes and token counts, every per-document NLL,
model asset hashes, checkpoint file hashes, tensor-state hashes, and all gate
inputs. The frozen manifest has SHA-256
`9e5d903b1a6cbcaafe0c216e36fd623fa55a31913c2e542f14cbcbca37d35ce9`;
the complete private raw report has SHA-256
`aaa95b69c39323cfbdd2d65ec4faaf7b2836c575719f7e0b65eed588d102b013`;
the independent replay has SHA-256
`0918d1caf97f456b7736009106639a5b120a360b36782300962237554c2de396`.

The source generator reproduced its train, development, and test files exactly.
The 50 canonical input-only documents had distinct hashes, no hash appeared in
earlier Compress evidence, and none exactly matched any of the 41,076 cached
WikiText train/test rows checked. These checks rule out exact-document reuse;
they do not establish semantic independence or natural-language diversity.

This result supports portable stored-size reduction and retained sampled
quality for one Pythia-70M model on this bounded synthetic panel. It establishes
no latency, peak-memory, energy, larger-model, tied-embedding, natural-corpus,
or state-of-the-art result. The next gate is a prospectively frozen evaluation
on an untouched natural corpus and a larger model with an ordinary untied
embedding. Tied embeddings require a separately validated shared-weight design.
