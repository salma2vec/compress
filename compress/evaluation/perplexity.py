"""Token-counted sliding-window likelihood for causal language models."""
import math
from typing import Optional

import torch


def _negative_log_likelihood(model, tokenizer, texts, max_length, stride, device):
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    stride = min(max_length - 1, max_length // 2) if stride is None else stride
    if not 1 <= stride < max_length:
        raise ValueError("stride must be in [1, max_length - 1]")
    device = torch.device(device) if device is not None else next(model.parameters(), next(model.buffers(), torch.empty(0))).device
    total_loss, total_tokens, total_bytes = 0., 0, 0
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for text in texts:
                tokens = tokenizer(text, return_tensors="pt", truncation=False)
                input_ids = tokens["input_ids"].to(device)
                seq_len = input_ids.shape[1]
                if seq_len < 2:
                    continue
                previous_end = 0
                for begin in range(0, seq_len, stride):
                    end = min(begin + max_length, seq_len)
                    ids = input_ids[:, begin:end]
                    labels = ids.clone()
                    new_tokens = end - previous_end
                    labels[:, :-new_tokens] = -100
                    kwargs = {}
                    if "attention_mask" in tokens:
                        mask = tokens["attention_mask"][:, begin:end].to(device)
                        labels[mask == 0] = -100
                        kwargs["attention_mask"] = mask
                    count = int((labels[:, 1:] != -100).sum())
                    if count:
                        loss = model(input_ids=ids, labels=labels, **kwargs).loss
                        if loss is None or not torch.isfinite(loss):
                            raise ValueError("Model returned non-finite likelihood")
                        total_loss += float(loss) * count
                        total_tokens += count
                    previous_end = end
                    if end == seq_len:
                        break
                total_bytes += len(text.encode("utf-8"))
    finally:
        model.train(training)
    if not total_tokens:
        raise ValueError("Evaluation contains no predictable tokens")
    return total_loss, total_tokens, total_bytes


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512,
                       stride: Optional[int] = None, device: Optional[str] = None) -> float:
    """Score every token after each document's first token exactly once.

    Windows retain context; stride is the number of new positions per window.
    Failures propagate instead of silently selecting an easier subset of data.
    """
    nll, count, _ = _negative_log_likelihood(model, tokenizer, texts, max_length, stride, device)
    try:
        return math.exp(nll / count)
    except OverflowError:
        return float("inf")


def compute_bits_per_byte(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """NLL in bits per UTF-8 byte over full documents (first tokens unscored)."""
    nll, _, byte_count = _negative_log_likelihood(model, tokenizer, texts, max_length, None, None)
    if not byte_count:
        raise ValueError("No UTF-8 bytes to score")
    return nll / byte_count / math.log(2)
