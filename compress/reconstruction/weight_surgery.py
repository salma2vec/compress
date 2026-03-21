import torch
import torch.nn as nn
from typing import Optional


def splice_weights(
    source: nn.Module,
    target: nn.Module,
    layer_names: list[str],
) -> nn.Module:
    """Copy specific layer weights from a source model into a target model.

    Useful for partially restoring high-importance layers after compression,
    or for grafting fine-tuned weights into a quantized base model.

    Only layers present in both source and target state dicts are copied;
    mismatched shape layers are skipped with a warning.

    Args:
        source: Model to copy weights from.
        target: Model to copy weights into. Modified in-place.
        layer_names: List of parameter names (as returned by state_dict()) to transfer.

    Returns:
        The modified target model.
    """
    source_state = source.state_dict()
    target_state = target.state_dict()

    updated = []
    skipped = []

    for name in layer_names:
        if name not in source_state:
            skipped.append((name, "not in source"))
            continue
        if name not in target_state:
            skipped.append((name, "not in target"))
            continue
        if source_state[name].shape != target_state[name].shape:
            skipped.append((name, f"shape mismatch: {source_state[name].shape} vs {target_state[name].shape}"))
            continue
        target_state[name] = source_state[name].clone()
        updated.append(name)

    if skipped:
        import warnings
        for name, reason in skipped:
            warnings.warn(f"splice_weights: skipping '{name}' ({reason})")

    target.load_state_dict(target_state)
    return target


def transplant_layers(
    source: nn.Module,
    target: nn.Module,
    layer_indices: list[int],
    layer_prefix: str = "transformer.h",
) -> nn.Module:
    """Copy entire transformer blocks from source to target by block index.

    Identifies all parameters belonging to the specified transformer blocks
    and delegates to splice_weights.

    Args:
        source: Source model.
        target: Target model. Modified in-place.
        layer_indices: Indices of transformer blocks to copy.
        layer_prefix: Name prefix identifying transformer block parameters.

    Returns:
        The modified target model.
    """
    source_state = source.state_dict()
    names_to_copy = [
        name for name in source_state
        if any(f"{layer_prefix}.{i}." in name for i in layer_indices)
    ]
    return splice_weights(source, target, names_to_copy)


def extract_layer_subset(model: nn.Module, layer_indices: list[int], layer_prefix: str = "transformer.h") -> dict[str, torch.Tensor]:
    """Extract state dict entries for a subset of transformer blocks.

    Args:
        model: Source model.
        layer_indices: Block indices to extract.
        layer_prefix: Prefix identifying transformer blocks.

    Returns:
        Filtered state dict containing only the specified blocks.
    """
    state = model.state_dict()
    return {
        name: tensor
        for name, tensor in state.items()
        if any(f"{layer_prefix}.{i}." in name for i in layer_indices)
    }
