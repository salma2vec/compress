import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class LayerStats:
    mean_magnitude: float = 0.0
    sparsity: float = 0.0       # fraction near-zero (< epsilon)
    variance: float = 0.0
    n_samples: int = 0


@dataclass
class ActivationProfile:
    layers: dict[str, LayerStats] = field(default_factory=dict)

    def head_stats(self, layer_idx: int, head_idx: int) -> LayerStats:
        key = f"layer.{layer_idx}.head.{head_idx}"
        return self.layers.get(key, LayerStats())

    def to_dict(self) -> dict[str, dict]:
        return {
            name: {
                "mean_magnitude": stats.mean_magnitude,
                "sparsity": stats.sparsity,
                "variance": stats.variance,
                "n_samples": stats.n_samples,
            }
            for name, stats in self.layers.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict]) -> "ActivationProfile":
        profile = cls()
        for name, d in data.items():
            if not all(math.isfinite(d[key]) for key in ("mean_magnitude", "sparsity", "variance", "n_samples")):
                raise ValueError("Activation statistics must be finite")
            if d["mean_magnitude"] < 0 or not 0 <= d["sparsity"] <= 1 or d["variance"] < 0 or d["n_samples"] < 1:
                raise ValueError("Invalid activation statistics")
            profile.layers[name] = LayerStats(
                mean_magnitude=d["mean_magnitude"],
                sparsity=d["sparsity"],
                variance=d["variance"],
                n_samples=d["n_samples"],
            )
        return profile


class ActivationRecorder:
    """Records activation statistics from forward passes through a model.

    Attaches forward hooks to attention and MLP submodules and accumulates
    mean magnitude, sparsity, and variance across a calibration corpus.

    Args:
        model: The transformer model to profile.
        epsilon: Threshold below which an activation is considered near-zero.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1e-3):
        self.model = model
        self.epsilon = epsilon
        self._hooks: list[Any] = []
        self._stats: dict[str, list[dict]] = {}

    def _make_hook(self, name: str):
        def hook(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            if not isinstance(t, torch.Tensor):
                return
            t = t.detach().float()
            if not t.numel() or not torch.isfinite(t).all():
                raise ValueError(f"Invalid activation in {name}")
            stats = self._stats.setdefault(name, dict(count=0, calls=0, abs_sum=0., sum=0., sum_sq=0., zeros=0))
            stats["count"] += t.numel()
            stats["calls"] += 1
            stats["abs_sum"] += t.abs().sum().item()
            stats["sum"] += t.sum().item()
            stats["sum_sq"] += t.square().sum().item()
            stats["zeros"] += (t.abs() < self.epsilon).sum().item()
        return hook

    def attach(self):
        """Idempotently hook weighted modules and attention/MLP containers."""
        if self._hooks:
            return
        for name, module in self.model.named_modules():
            weight = getattr(module, "weight", None)
            if (isinstance(weight, torch.Tensor) and weight.ndim >= 2) or any(k in name for k in ["attn", "attention", "mlp", "feed_forward"]):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self):
        self._stats.clear()

    def record(self, inputs: list[str], tokenizer, n_batches: int = 10) -> ActivationProfile:
        """Record element-weighted statistics, failing visibly on broken samples."""
        if n_batches < 1 or not inputs:
            raise ValueError("Calibration requires positive n_batches and nonempty inputs")
        self.reset()
        self.attach()
        training = self.model.training
        self.model.eval()
        device = next(self.model.parameters(), torch.empty(0)).device
        try:
            with torch.no_grad():
                for text in inputs[:n_batches]:
                    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=False)
                    tokens = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in tokens.items()}
                    self.model(**tokens)
        finally:
            self.detach()
            self.model.train(training)
        profile = ActivationProfile()
        for name, stats in self._stats.items():
            n = stats["count"]
            profile.layers[name] = LayerStats(
                mean_magnitude=stats["abs_sum"] / n,
                sparsity=stats["zeros"] / n,
                variance=max(0., stats["sum_sq"] / n - (stats["sum"] / n) ** 2),
                n_samples=stats["calls"],
            )
        if not profile.layers:
            raise ValueError("No activations recorded for supported modules")
        return profile
