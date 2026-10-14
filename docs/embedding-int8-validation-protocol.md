# Portable INT8 embedding validation: prospective local protocol

Version 1, 2026-10-14. Freeze the implementation, source revision, cached-asset
hashes and a new paragraph manifest before evaluating any model. This study asks
whether extending the existing portable rowwise-INT8 checkpoint backend to the
input embedding yields a useful saved model. It is basic INT8 engineering, not a
new quantization algorithm, and it makes no latency or peak-memory claim because
the backend dequantizes for ordinary PyTorch operations.

Use only the pinned cached Pythia-70M safetensors/tokenizer and WikiText-2 test
Parquet already recorded by the earlier study. The model has an independent
50,304 by 512 input embedding (`tie_word_embeddings=false`). Compare exactly:

1. FP32 reference.
2. Every supported linear module in rowwise INT8, input embedding left FP32.
3. The same rowwise-INT8 linear policy plus the input embedding in rowwise INT8.

The third method is the only new measured intervention. No calibration or
ranking is performed. Quantization and evaluation run on CPU with one Torch and
BLAS thread. The checkpoint must retain the embedding's declared options and
must reject incompatible archives before mutating the caller's model.

Create a fresh article-grouped panel with fixed seed 1043. Exclude every complete
test article exposed by either the original 16-paragraph pilot or the 32-paragraph
allocation study. Also exclude exact duplicates of every calibration paragraph
from those studies. Select two eligible 128–512-token paragraphs from each of
eight remaining articles. If fewer than eight article groups remain, stop before
loading model weights; do not fall back after inspecting losses. The old exposed
panel may be used only as an explicitly labeled diagnostic fallback under a new
protocol.

Score every target after each paragraph's first token exactly once, using context
256 and stride 128. Retain paragraph/article IDs, hashes, loss sums and token
counts. For all three methods record actual persistent parameter-plus-buffer
bytes. Save and reload each method with safetensors, require identical state
tensor hashes and exact CPU logits on a declared check window, independently
verify shifted-token loss against the model loss, and remove the temporary
checkpoint only after recording its file hash. Record the CPU-only device and
fallback contract explicitly.

The practical gate for linear-plus-embedding INT8 requires all correctness gates,
exactly 71,062,592 persistent tensor bytes, and perplexity at most 1.05 times the
fresh-panel FP32 perplexity. Report the linear-only INT8 result alongside it, but
do not use that comparator to weaken the gate. A pass is a one-model, one-corpus
portable-storage result; it is not evidence of low-bit execution speed, broad
model generalization, or superiority to AWQ/GPTQ. A failure remains a completed
result and does not alter the frozen mixed-allocation study or its gates.

Related primary work: [AWQ](https://arxiv.org/abs/2306.00978v6) uses activation-
informed channel scaling and hardware-aware weight packing, neither of which is
implemented here. [CASA](https://arxiv.org/abs/2609.25916) and
[MixQuant](https://arxiv.org/abs/2607.23047) show why isolated scalar layer scores
can miss cross-layer and upstream-quantization interactions. Their reported
results are context for this study's limits and are not borrowed as evidence.
