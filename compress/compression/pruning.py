import torch
import torch.nn as nn


class HeadPruner:
    """Structured pruning of attention layers ranked by importance.

    Zeroes out the weights (and biases) of the lowest-scoring attention
    layers, effectively removing their contribution to the forward pass.
    This is a structured, magnitude-based pruning strategy — pruned layers
    retain their parameter count but contribute zero activations.

    For deployment efficiency, pair with a subsequent weight-dropping pass
    or use sparse kernel support to skip zeroed layers at inference time.

    Args:
        keep_fraction: Fraction of attention layers to retain (0.0–1.0).
            E.g., 0.7 keeps the top 70% by importance.
    """

    def __init__(self, keep_fraction: float = 0.7):
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction
        self.pruned_heads: dict[str, list[int]] = {}

    def prune_heads(self, model: nn.Module, importance: dict[str, float]) -> nn.Module:
        """Zero out the bottom (1 - keep_fraction) of attention layers by importance.

        Args:
            model: The model to prune. Modified in-place.
            importance: Dict mapping layer names to importance scores (0–1).

        Returns:
            The pruned model.
        """
        attn_layers = {
            k: v for k, v in importance.items()
            if "attn" in k or "attention" in k
        }
        if not attn_layers:
            return model

        sorted_layers = sorted(attn_layers.items(), key=lambda x: x[1])
        n_to_prune = int(len(sorted_layers) * (1.0 - self.keep_fraction))
        to_prune = set(k for k, _ in sorted_layers[:n_to_prune])

        for name, module in model.named_modules():
            if name not in to_prune:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.zero_()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            self.pruned_heads[name] = []

        return model

    def pruning_report(self) -> dict:
        """Return a summary of pruned layers."""
        return {
            "n_pruned": len(self.pruned_heads),
            "pruned_layers": list(self.pruned_heads.keys()),
        }
