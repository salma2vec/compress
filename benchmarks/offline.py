"""Reproducible CPU integration benchmark on a synthetic causal prediction task.

This is a functional toy experiment, not evidence of pretrained LLM quality.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from compress.compression.mixed_precision import MixedPrecisionConfig
from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import QuantizationConfig
from compress.evaluation.perplexity import compute_perplexity
from compress.profiling.activation_recorder import ActivationRecorder
from compress.profiling.importance import ImportanceScorer


class Tokenizer:
    def __call__(self, text, max_length=128, truncation=False, **kwargs):
        ids = [ord(c) - 97 for c in text]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TinyCausalLM(nn.Module):
    def __init__(self, width=32, vocabulary=16):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.attn_out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.head = nn.Linear(width, vocabulary)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        q, k, v = self.q(x)[:, None], self.k(x)[:, None], self.v(x)[:, None]
        x = x + self.attn_out(torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)[:, 0])
        x = self.norm(x + self.mlp(x))
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def corpus(seed, n, length):
    rng = random.Random(seed)
    result = []
    for _ in range(n):
        start = rng.randrange(16)
        # Shared learnable transition rule; splits differ in sequence length.
        result.append(''.join(chr(97 + (start + i) % 16) for i in range(length)))
    return result


def run(output, seed=42, steps=100):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model, tokenizer = TinyCausalLM(), Tokenizer()
    training = corpus(seed, 32, 24)
    calibration = corpus(seed + 1, 16, 29)
    evaluation = corpus(seed + 2, 16, 37)
    assert not set(training) & set(calibration) and not set(calibration) & set(evaluation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    ids = torch.cat([tokenizer(t)["input_ids"] for t in training])
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
    model.eval()
    profile = ActivationRecorder(model).record(calibration, tokenizer, n_batches=len(calibration))
    scores = ImportanceScorer(profile).layer_importance()
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    scores = {name: scores[name] for name in names}
    mixed = MixedPrecisionConfig.from_importance(scores).layer_dtypes
    # Shuffle only among equal-shape layers to hold stored bytes exactly fixed.
    shuffled = dict(mixed)
    groups = {}
    for name in names:
        groups.setdefault(tuple(model.get_submodule(name).weight.shape), []).append(name)
    rng = random.Random(seed)
    for group in groups.values():
        precisions = [mixed[name] for name in group]
        rng.shuffle(precisions)
        shuffled.update(zip(group, precisions))
    methods = {"fp32": None, "uniform_int8": {name: "int8" for name in names},
               "uniform_int4": {name: "int4" for name in names}, "mixed": mixed, "mixed_shuffled": shuffled}
    report = {"schema_version": 1, "task": "synthetic cyclic-symbol next-token prediction",
              "scope": "CPU integration experiment; no pretrained LLM or GPU claim",
              "seed": seed, "training_steps": steps, "final_training_loss": float(loss.detach()),
              "torch": torch.__version__, "python": platform.python_version(), "device": "cpu", "threads": 1,
              "split_sha256": {name: hashlib.sha256('\n'.join(texts).encode()).hexdigest()
                               for name, texts in [("training", training), ("calibration", calibration), ("evaluation", evaluation)]},
              "importance": scores, "results": []}
    heldout = tokenizer(evaluation[0])["input_ids"]
    with torch.no_grad():
        reference = model(heldout).logits
    with tempfile.TemporaryDirectory() as tmp:
        for name, assignment in methods.items():
            candidate = copy.deepcopy(model)
            if assignment is not None:
                candidate = pack_model(candidate, QuantizationConfig(assignment))
                checkpoint = Path(tmp) / (name + '.safetensors')
                save_packed(candidate, checkpoint)
                candidate = load_packed(TinyCausalLM(), checkpoint).eval()
                checkpoint_bytes = checkpoint.stat().st_size
            else:
                checkpoint = Path(tmp) / 'fp32.safetensors'
                save_file(candidate.state_dict(), str(checkpoint))
                checkpoint_bytes = checkpoint.stat().st_size
            with torch.no_grad():
                logits = candidate(heldout).logits
                kl = torch.nn.functional.kl_div(logits.log_softmax(-1), reference.softmax(-1), reduction='sum') / heldout.numel()
                timings = []
                for i in range(13):
                    start = time.perf_counter()
                    candidate(heldout)
                    if i >= 3:
                        timings.append(time.perf_counter() - start)
            report["results"].append({"method": name, "assignments": assignment,
                                      "perplexity": compute_perplexity(candidate, tokenizer, evaluation, max_length=32, stride=16),
                                      "reference_kl_per_token": max(0., float(kl)),
                                      "persistent_tensor_bytes": tensor_bytes(candidate), "checkpoint_bytes": checkpoint_bytes,
                                      "prefill_median_seconds": statistics.median(timings), "prefill_samples_seconds": timings})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/offline.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    report = run(args.output, args.seed, args.steps)
    for row in report['results']:
        print(f"{row['method']}: perplexity={row['perplexity']:.4f}, tensor_bytes={row['persistent_tensor_bytes']}")
