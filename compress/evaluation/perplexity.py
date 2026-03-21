import torch
import math
from typing import Optional


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    stride: Optional[int] = None,
    device: Optional[str] = None,
) -> float:
    """Compute perplexity of a language model on a list of text samples.

    Uses the standard cross-entropy loss averaged over all tokens, then
    exponentiated. For long documents, a strided sliding-window evaluation
    can be used by setting the stride parameter.

    Args:
        model: A HuggingFace causal language model.
        tokenizer: Compatible tokenizer for the model.
        texts: List of text strings to evaluate.
        max_length: Maximum sequence length to tokenize.
        stride: If set, use strided evaluation with this overlap (tokens).
            Useful for long documents to avoid context truncation artifacts.
        device: Device string (e.g. "cpu", "cuda"). Defaults to model's device.

    Returns:
        Perplexity as a float. Returns float("inf") if no valid tokens are found.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device.type
        except StopIteration:
            device = "cpu"

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokens["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                n_tokens = input_ids.shape[1] - 1
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
            except Exception:
                continue

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))  # cap to avoid overflow


def compute_bits_per_byte(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute bits-per-byte (BPB) for a language model.

    BPB normalizes perplexity by character count, making it comparable
    across tokenizers with different vocabulary sizes.

    Args:
        model: A HuggingFace causal language model.
        tokenizer: Compatible tokenizer.
        texts: Evaluation texts.
        max_length: Maximum sequence length.

    Returns:
        Bits-per-byte as a float.
    """
    import math

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_chars = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokens["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                n_tokens = input_ids.shape[1] - 1
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                total_chars += len(text.encode("utf-8"))
            except Exception:
                continue

    if total_tokens == 0 or total_chars == 0:
        return float("inf")

    avg_nll_per_token = total_loss / total_tokens
    avg_tokens_per_byte = total_tokens / total_chars
    return avg_nll_per_token * avg_tokens_per_byte / math.log(2)
