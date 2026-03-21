import torch
import torch.nn as nn
from typing import Optional


class LayerDropDistillation:
    """Select a subset of transformer layers to keep based on activation importance.

    Layer dropping is a form of structural distillation: a smaller student model
    is constructed by retaining only the most important layers from a teacher.
    Importance is estimated from activation profiles; layers with consistently
    high-magnitude, low-sparsity activations are prioritized.

    The selected layer indices can be used to construct a student model via
    standard HuggingFace config manipulation or manual weight surgery.

    Args:
        keep_fraction: Fraction of layers to retain. Must be in (0, 1].
    """

    def __init__(self, keep_fraction: float = 0.75):
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction

    def select_layers_to_keep(
        self,
        importance: dict[str, float],
        n_layers: int,
    ) -> list[int]:
        """Return indices of the layers to retain in the student model.

        Layer importance is aggregated across all profiled submodules that
        belong to each transformer block (identified by block index in the
        layer name).

        Args:
            importance: Dict mapping layer name to importance score in [0, 1].
            n_layers: Total number of transformer blocks in the model.

        Returns:
            Sorted list of layer indices to keep.
        """
        n_keep = max(1, int(n_layers * self.keep_fraction))

        if not importance:
            return list(range(n_keep))

        # Aggregate importance scores per transformer block index
        layer_scores: dict[int, float] = {}
        for name, score in importance.items():
            for i in range(n_layers):
                if f".{i}." in name or name.endswith(f".{i}"):
                    layer_scores[i] = layer_scores.get(i, 0.0) + score
                    break

        if not layer_scores:
            return list(range(n_keep))

        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        kept = sorted(idx for idx, _ in sorted_layers[:n_keep])
        return kept

    def compute_layer_scores(
        self,
        importance: dict[str, float],
        n_layers: int,
    ) -> dict[int, float]:
        """Return aggregated importance score per transformer block.

        Args:
            importance: Dict mapping layer name to importance score.
            n_layers: Total number of transformer blocks.

        Returns:
            Dict mapping block index to aggregated importance score.
        """
        layer_scores: dict[int, float] = {}
        for name, score in importance.items():
            for i in range(n_layers):
                if f".{i}." in name or name.endswith(f".{i}"):
                    layer_scores[i] = layer_scores.get(i, 0.0) + score
                    break
        return layer_scores
