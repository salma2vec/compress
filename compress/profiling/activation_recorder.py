import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any


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
            # Handle tuple outputs (e.g. attention layers return (attn_out, weights))
            if isinstance(output, tuple):
                t = output[0].detach().float()
            else:
                t = output.detach().float()

            if name not in self._stats:
                self._stats[name] = []
            self._stats[name].append({
                "mean_mag": t.abs().mean().item(),
                "sparsity": (t.abs() < self.epsilon).float().mean().item(),
                "variance": t.var().item() if t.numel() > 1 else 0.0,
            })
        return hook

    def attach(self):
        """Attach forward hooks to all attention and MLP submodules."""
        for name, module in self.model.named_modules():
            if any(k in name for k in ["attn", "attention", "mlp", "feed_forward"]):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def detach(self):
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        """Clear accumulated statistics."""
        self._stats.clear()

    def record(self, inputs: list[str], tokenizer, n_batches: int = 10) -> ActivationProfile:
        """Run forward passes and accumulate activation statistics.

        Args:
            inputs: List of text strings to use as calibration inputs.
            tokenizer: HuggingFace tokenizer compatible with the model.
            n_batches: Maximum number of inputs to process.

        Returns:
            ActivationProfile with per-layer statistics.
        """
        self.reset()
        self.attach()
        self.model.eval()

        with torch.no_grad():
            for text in inputs[:n_batches]:
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=False,
                )
                try:
                    self.model(**tokens)
                except Exception:
                    # Skip samples that cause errors (e.g. empty sequences)
                    continue

        self.detach()

        profile = ActivationProfile()
        for name, stats_list in self._stats.items():
            if not stats_list:
                continue
            profile.layers[name] = LayerStats(
                mean_magnitude=sum(s["mean_mag"] for s in stats_list) / len(stats_list),
                sparsity=sum(s["sparsity"] for s in stats_list) / len(stats_list),
                variance=sum(s["variance"] for s in stats_list) / len(stats_list),
                n_samples=len(stats_list),
            )
        return profile
