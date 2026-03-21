import time
import torch
from typing import Optional


def measure_latency(
    model,
    tokenizer,
    prompt: str = "The quick brown fox",
    n_tokens: int = 50,
    n_runs: int = 5,
    warmup_runs: int = 2,
    device: Optional[str] = None,
) -> dict:
    """Measure token generation latency for a causal language model.

    Runs the model for n_runs timed forward passes and returns summary
    statistics. Includes warmup runs to ensure JIT compilation and cache
    effects are stable before timing begins.

    Args:
        model: A HuggingFace causal language model with .generate().
        tokenizer: Compatible tokenizer.
        prompt: Input prompt string.
        n_tokens: Number of new tokens to generate per run.
        n_runs: Number of timed runs for statistics.
        warmup_runs: Number of untimed warmup passes.
        device: Device string. Defaults to model's current device.

    Returns:
        Dict with keys:
            - mean_latency_s: Mean wall-clock time per generation.
            - std_latency_s: Standard deviation of latency.
            - min_latency_s: Minimum latency observed.
            - tokens_per_sec: Throughput estimate (n_tokens / mean_latency).
            - n_runs: Number of timed runs.
    """
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            try:
                model.generate(**tokens, max_new_tokens=n_tokens, do_sample=False)
            except Exception:
                pass

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            try:
                model.generate(**tokens, max_new_tokens=n_tokens, do_sample=False)
            except Exception:
                pass
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    if not latencies:
        return {
            "mean_latency_s": float("nan"),
            "std_latency_s": float("nan"),
            "min_latency_s": float("nan"),
            "tokens_per_sec": 0.0,
            "n_runs": 0,
        }

    mean_lat = sum(latencies) / len(latencies)
    variance = sum((x - mean_lat) ** 2 for x in latencies) / max(len(latencies) - 1, 1)
    std_lat = variance ** 0.5

    return {
        "mean_latency_s": mean_lat,
        "std_latency_s": std_lat,
        "min_latency_s": min(latencies),
        "tokens_per_sec": n_tokens / mean_lat if mean_lat > 0 else 0.0,
        "n_runs": n_runs,
    }


def measure_prefill_latency(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 256,
    n_runs: int = 3,
) -> dict:
    """Measure prefill (prompt processing) latency without generation.

    Useful for benchmarking encoder models or the prefill phase of
    decoder models independently from token generation.

    Args:
        model: Model to benchmark.
        tokenizer: Compatible tokenizer.
        texts: Input prompts.
        max_length: Maximum tokenized length.
        n_runs: Timed runs per input.

    Returns:
        Dict with mean_latency_s, tokens_per_sec, and total_tokens.
    """
    model.eval()
    latencies = []
    total_tokens = 0

    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        n_tok = tokens["input_ids"].shape[1]

        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(**tokens)
            latencies.append(time.perf_counter() - start)
            total_tokens += n_tok

    if not latencies:
        return {"mean_latency_s": float("nan"), "tokens_per_sec": 0.0, "total_tokens": 0}

    mean_lat = sum(latencies) / len(latencies)
    return {
        "mean_latency_s": mean_lat,
        "tokens_per_sec": (total_tokens / len(texts)) / mean_lat if mean_lat > 0 else 0.0,
        "total_tokens": total_tokens,
    }
