import torch
import torch.nn as nn
from .activation_recorder import ActivationProfile


class ImportanceScorer:
    """Computes per-layer importance scores from an ActivationProfile.

    Importance is a composite of activation magnitude and density. High-magnitude,
    low-sparsity layers are scored near 1.0; dormant layers score near 0.0.

    Args:
        profile: An ActivationProfile collected via ActivationRecorder.
    """

    def __init__(self, profile: ActivationProfile):
        self.profile = profile

    def layer_importance(self) -> dict[str, float]:
        """Score each layer 0–1. Higher = more important (preserve at higher precision).

        The score combines mean activation magnitude with activation density
        (1 - sparsity). Both factors are necessary: a layer can have high magnitude
        but near-total sparsity, making it effectively dormant.

        Returns:
            Dict mapping layer name to normalized importance score in [0, 1].
        """
        scores: dict[str, float] = {}
        for name, stats in self.profile.layers.items():
            importance = stats.mean_magnitude * (1.0 - stats.sparsity)
            scores[name] = max(0.0, importance)

        # Normalize to [0, 1]
        if scores:
            max_v = max(scores.values()) or 1.0
            scores = {k: min(v / max_v, 1.0) for k, v in scores.items()}
        return scores

    def head_importance(self, model: nn.Module = None, calibration_inputs=None) -> dict[str, float]:
        """Return attention-module activation scores (not individual heads).

        Kept for API compatibility. Gradient/head-level scoring is not implemented;
        arguments requesting that behavior fail rather than being ignored.
        """
        if model is not None or calibration_inputs is not None:
            raise NotImplementedError("Gradient or individual-head scoring is not implemented")
        scores: dict[str, float] = {}
        for name, stats in self.profile.layers.items():
            if "attn" in name or "attention" in name:
                scores[name] = max(0.0, stats.mean_magnitude * (1.0 - stats.sparsity))

        if scores:
            max_v = max(scores.values()) or 1.0
            scores = {k: min(v / max_v, 1.0) for k, v in scores.items()}
        return scores

    def recommended_dtype(self, layer_name: str) -> str:
        """Return the recommended quantization dtype for a given layer.

        Thresholds:
          - score > 0.8 → fp16 (preserve fidelity)
          - score > 0.5 → int8 (moderate compression)
          - score ≤ 0.5 → int4 (aggressive compression)

        Args:
            layer_name: The name of the layer as it appears in the ActivationProfile.

        Returns:
            One of "fp16", "int8", or "int4".
        """
        scores = self.layer_importance()
        s = scores.get(layer_name, 0.5)
        if s > 0.8:
            return "fp16"
        elif s > 0.5:
            return "int8"
        else:
            return "int4"

    def summary(self) -> dict:
        """Return a summary of importance distribution across all layers."""
        scores = self.layer_importance()
        if not scores:
            return {"n_layers": 0, "mean_importance": 0.0, "dtype_breakdown": {}}

        values = list(scores.values())
        n_fp16 = sum(1 for v in values if v > 0.8)
        n_int8 = sum(1 for v in values if 0.5 < v <= 0.8)
        n_int4 = sum(1 for v in values if v <= 0.5)

        return {
            "n_layers": len(scores),
            "mean_importance": sum(values) / len(values),
            "min_importance": min(values),
            "max_importance": max(values),
            "dtype_breakdown": {"fp16": n_fp16, "int8": n_int8, "int4": n_int4},
        }
