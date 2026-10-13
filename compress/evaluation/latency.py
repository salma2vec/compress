"""Synchronized wall-clock latency with actual generated token counts."""
import statistics
import time

import torch


def _device(model, device=None):
    return torch.device(device) if device is not None else next(model.parameters(), next(model.buffers(), torch.empty(0))).device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _tokens(tokenizer, prompt, device, **kwargs):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenizer(prompt, return_tensors="pt", **kwargs).items()}


def measure_latency(model, tokenizer, prompt="The quick brown fox", n_tokens=50,
                    n_runs=5, warmup_runs=2, device=None):
    if n_tokens < 1 or n_runs < 1 or warmup_runs < 0:
        raise ValueError("Token/run counts must be positive; warmups nonnegative")
    device = _device(model, device)
    tokens = _tokens(tokenizer, prompt, device)
    training = model.training
    model.eval()
    latencies, generated = [], []
    try:
        with torch.no_grad():
            for i in range(warmup_runs + n_runs):
                _sync(device)
                start = time.perf_counter()
                output = model.generate(**tokens, max_new_tokens=n_tokens, do_sample=False)
                _sync(device)
                elapsed = time.perf_counter() - start
                if i >= warmup_runs:
                    latencies.append(elapsed)
                    generated.append(int(output.shape[-1] - tokens["input_ids"].shape[-1]))
    finally:
        model.train(training)
    return {"mean_latency_s": statistics.mean(latencies),
            "std_latency_s": statistics.stdev(latencies) if n_runs > 1 else 0.,
            "min_latency_s": min(latencies), "tokens_per_sec": sum(generated) / sum(latencies),
            "n_runs": n_runs, "generated_tokens": generated, "latencies_s": latencies}


def measure_prefill_latency(model, tokenizer, texts, max_length=256, n_runs=3):
    if not texts or n_runs < 1:
        raise ValueError("Nonempty texts and positive run count required")
    device = _device(model)
    training = model.training
    model.eval()
    latencies, total_tokens = [], 0
    try:
        with torch.no_grad():
            for text in texts:
                tokens = _tokens(tokenizer, text, device, truncation=True, max_length=max_length)
                model(**tokens)  # untimed warmup
                for _ in range(n_runs):
                    _sync(device)
                    start = time.perf_counter()
                    model(**tokens)
                    _sync(device)
                    latencies.append(time.perf_counter() - start)
                    total_tokens += int(tokens.get("attention_mask", torch.ones_like(tokens["input_ids"])).sum())
    finally:
        model.train(training)
    return {"mean_latency_s": statistics.mean(latencies), "tokens_per_sec": total_tokens / sum(latencies),
            "total_tokens": total_tokens, "latencies_s": latencies}
