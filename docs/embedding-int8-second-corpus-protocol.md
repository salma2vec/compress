# Portable INT8 embedding validation on a second domain

## Frozen question

Does the already implemented rowwise-INT8 embedding path retain its measured
storage and sampled-quality behavior on a corpus that was never used by the
earlier Compress studies? This is a second-domain engineering check of the same
Pythia-70M artifact and the same three saved methods. It is not a new
quantization algorithm.

The corpus is the complete 50-record test split from the local `patchy`
incident-remediation project. It was deterministically generated on 2026-02-15
from the MIT-licensed generator at commit
`3813eae295e9dd9b411fedba10df0791ee66e3a8`, using seed 0 after generating
200 training and 50 development records. A pre-inference check reproduced all
three split files byte for byte from that source. The test file has SHA-256
`50d447a3a4d00ea3996b889ec3e3a2e1ed2807a6bb9a786eb0010d1d492ca70e`.

This corpus is synthetic, small, and template-generated. It can detect a gross
domain-specific regression in the portable checkpoint path, but it cannot
establish generalization to a second natural corpus, another model, or a wider
distribution of language.

## Frozen inputs

Use all 50 test records in file order. Canonicalize only fields that an incident
remediation model would receive: `title`, `narrative`, `logs`, `metrics`,
`recent_deploys`, and `config_diff`. Exclude action labels, evidence labels, and
root-cause labels. Prefix scalar fields and render list entries as ordered
hyphen bullets. Render an empty deployment or configuration string as
`(none)`.

Before tokenization, require 50 distinct incident identifiers and 50 distinct
canonical document hashes. Compare those hashes with every cached WikiText-2
raw train and test row and require zero exact matches. A pre-source-freeze scan
at Compress commit `4db95a451e05a5a80776ba58891b7306144157ba`
also requires no corpus-file or canonical-document hash in prior public or
private Compress evidence. Freeze every document hash and token count in the
manifest before model-weight loading.

Use cached `EleutherAI/pythia-70m` assets only, with their exact previously
published hashes. Run on CPU with one Torch and BLAS thread, eager attention,
context 256, stride 128, seed 42, and no accelerator fallback. Compare exactly:

1. FP32.
2. Every supported linear module in rowwise INT8, input embedding left FP32.
3. The same linear policy plus the ordinary untied input embedding in rowwise
   INT8.

All methods must score every predicted target once. Save each method to
safetensors, record the checkpoint file hash and persistent tensor bytes,
rebuild the architecture, reload, and require identical state-tensor hashes and
exact CPU logits on the frozen check window. Independently check the shifted
causal loss. Delete temporary checkpoints only after recording their hashes.

## Frozen acceptance rule

The linear-plus-embedding candidate passes only if every correctness check
passes, its persistent tensor count is exactly 71,062,592 bytes, and its
perplexity is at most 1.05 times FP32 on the complete frozen panel. Record and
publish the outcome even if it fails. Do not alter documents, thresholds,
precision assignments, or evaluation settings after observing any model
result. No retry is permitted for a quality failure.

## Related primary work

Guan et al. study post-training row-wise quantization specifically for large
embedding tables ([arXiv:1911.02079](https://arxiv.org/abs/1911.02079)). AWQ
uses activation information and equivalent channel scaling for low-bit LLM
weights ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978), current v6).
Han's recent measured map reports that module-level sensitivity and embedding
protection heuristics do not transfer uniformly across text-embedder families
([arXiv:2609.16391](https://arxiv.org/abs/2609.16391)). These papers motivate
measuring stored bytes and end-task behavior rather than assuming an embedding
policy transfers. This protocol reproduces none of their methods or reported
results.
