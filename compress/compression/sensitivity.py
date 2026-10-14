"""Calibration-only, single-module intervention scores and exact-shape plans.

This is a direct perturbation baseline, not a new quantization method. Isolated
scores do not account for interactions among simultaneously quantized modules.
"""

import copy
import math

import torch
from torch import nn

from .mixed_precision import MixedPrecisionConfig
from .packed import PackedLinear, tensor_bytes


def shifted_loss(model, input_ids):
    """Compute the shifted-token objective directly instead of model labels."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("Calibration inputs must be one sequence with at least two tokens")
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        value = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1), reduction="sum",
        )
    if not bool(torch.isfinite(value)):
        raise ValueError("Nonfinite calibration loss")
    return float(value), input_ids.shape[1] - 1


def module_sensitivity(model, calibration_ids, *, output_head_names=("lm_head",)):
    """Measure NF4 minus INT8 loss with every other module left untouched.

    The caller supplies a full-precision model and preselected calibration-only
    tokens. Every replacement is restored even if inference fails. Output heads
    remain protected and therefore do not need a sensitivity measurement.
    """
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)]
    if not calibration_ids or not modules or any(isinstance(m, PackedLinear) for m in model.modules()):
        raise ValueError("Require a full-precision model and nonempty calibration data")
    names = {name for name, _ in modules}
    if not set(output_head_names) <= names:
        raise ValueError("Declared output head is not a linear module")
    if "" in names:
        raise ValueError("The model must contain named linear modules")
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    results = {}
    try:
        for name, original in modules:
            if name in output_head_names:
                continue
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            measurements = {}
            try:
                for precision in ("int8", "int4"):
                    # Match the CPU packing used by archive evaluation, even
                    # when calibration inference runs on an accelerator.
                    packed = PackedLinear(copy.deepcopy(original).cpu(), precision)
                    packed = packed.to(original.weight.device)
                    setattr(parent, leaf, packed)
                    per_sequence = [shifted_loss(model, ids) for ids in calibration_ids]
                    count = sum(n for _, n in per_sequence)
                    measurements[precision] = {
                        "nll": sum(loss for loss, _ in per_sequence), "tokens": count,
                        "tensor_bytes": tensor_bytes(packed),
                    }
                    setattr(parent, leaf, original)
            finally:
                setattr(parent, leaf, original)
            int8, nf4 = measurements["int8"], measurements["int4"]
            if int8["tokens"] != nf4["tokens"] or int8["tokens"] < 1:
                raise ValueError("Interventions did not score identical targets")
            cost = int8["tensor_bytes"] - nf4["tensor_bytes"]
            if cost <= 0:
                raise ValueError("INT8 promotion must have a positive storage cost")
            benefit = (nf4["nll"] - int8["nll"]) / int8["tokens"]
            results[name] = {
                "int8": int8, "int4": nf4, "nll_benefit": benefit,
                "additional_bytes": cost, "benefit_per_byte": benefit / cost,
            }
    finally:
        # Calling train(parent_flag) would erase intentionally mixed child modes.
        for module, training in training_flags:
            module.training = training
    return results


def equal_shape_plan(model, scores, *, output_head_names=("lm_head",)):
    """Promote the higher-scoring half within each exact storage-cost group.

    Equal weight/bias shapes give identical packed byte costs. This restricted
    allocation matches the historical half-per-group budget; it is not a global
    knapsack optimizer. All declared output heads are always INT8.
    """
    groups = {}
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        names.add(name)
        if name in output_head_names:
            continue
        if name not in scores or not math.isfinite(scores[name]):
            raise ValueError(f"Missing or nonfinite score: {name}")
        key = (tuple(module.weight.shape), module.bias is not None,
               None if module.bias is None else module.bias.dtype)
        groups.setdefault(key, []).append(name)
    if not names or not set(output_head_names) <= names:
        raise ValueError("Declared output head is not a linear module")
    assignment = {name: "int8" for name in output_head_names}
    for group in groups.values():
        ranked = sorted(group, key=lambda name: (-scores[name], name))
        assignment.update({name: "int8" if i < (len(ranked) + 1) // 2 else "int4"
                           for i, name in enumerate(ranked)})
    return MixedPrecisionConfig(assignment, output_head_names=tuple(output_head_names))
