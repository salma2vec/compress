"""Portable weight-only storage; dequantizes for ordinary PyTorch matmul.

This backend reduces stored tensor bytes, not arithmetic cost or peak memory.
It intentionally makes no low-bit kernel throughput claim.
"""
import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from .quantization import Int8Quantizer, NF4Quantizer, QuantizationConfig


class PackedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, dtype: str):
        super().__init__()
        if dtype not in {"fp16", "int8", "int4"}:
            raise ValueError("Unsupported dtype")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.precision = dtype
        weight = linear.weight.detach().float()
        if dtype == "fp16":
            self.register_buffer("packed_weight", weight.half())
            self.register_buffer("scale", torch.empty(0, device=weight.device))
        elif dtype == "int8":
            quantized, scale, _ = Int8Quantizer.quantize_tensor(weight)
            self.register_buffer("packed_weight", quantized)
            self.register_buffer("scale", scale)
        else:
            # Per-output-row NF4 normalization; two indices per uint8.
            scale = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            normalized = weight / scale[:, None]
            levels = NF4Quantizer.NF4_LEVELS.to(weight.device)
            indices = torch.bucketize(normalized.contiguous(), (levels[:-1] + levels[1:]) / 2).to(torch.uint8).flatten()
            if indices.numel() % 2:
                indices = torch.cat((indices, indices.new_zeros(1)))
            self.register_buffer("packed_weight", indices[::2] | (indices[1::2] << 4))
            self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def dequantize(self):
        if self.precision == "fp16":
            return self.packed_weight.float()
        if self.precision == "int8":
            return self.packed_weight.float() * self.scale[:, None]
        indices = torch.stack((self.packed_weight & 15, self.packed_weight >> 4), dim=-1).flatten()
        indices = indices[:self.in_features * self.out_features].reshape(self.out_features, self.in_features)
        levels = NF4Quantizer.NF4_LEVELS.to(indices.device)
        return levels[indices.long()] * self.scale[:, None]

    def forward(self, inputs):
        weight = self.dequantize().to(inputs.dtype)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return torch.nn.functional.linear(inputs, weight, bias)


def pack_model(model: nn.Module, config: QuantizationConfig):
    """Replace ordinary nn.Linear modules in place; reject tied/shared weights.

    Some architectures directly access child .weight (e.g. MultiheadAttention).
    Such modules cannot use this portable backend; the caller must exclude them.
    """
    QuantizationConfig(dict(config.layer_dtypes), config.default_dtype)
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not modules:
        raise ValueError("No nn.Linear modules found")
    parameter_ids = [id(p) for _, p in model.named_parameters(remove_duplicate=False)]
    for name, module in modules:
        if parameter_ids.count(id(module.weight)) > 1:
            raise ValueError(f"Tied/shared linear weight is unsupported: {name}")
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(parent, nn.MultiheadAttention):
            raise ValueError("MultiheadAttention accesses out_proj.weight directly; use fake quantization")
    for name, module in modules:
        packed = PackedLinear(module, config.layer_dtypes.get(name, config.default_dtype))
        if not name:
            return packed
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, packed)
    return model


def tensor_bytes(model):
    """Persistent parameter + buffer bytes, excluding temporary dequantization."""
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def save_packed(model, path):
    spec = {name: module.precision for name, module in model.named_modules() if isinstance(module, PackedLinear)}
    if not spec:
        raise ValueError("Model has no packed layers")
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}
    save_file(state, str(path), metadata={"format_version": "1", "layers": json.dumps(spec)})


def load_packed(model, path):
    """Restore into a caller-created matching architecture without pickle objects."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format_version") != "1":
        raise ValueError("Unsupported packed checkpoint version")
    layers = json.loads(metadata["layers"])
    expected = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    if set(layers) != expected:
        raise ValueError("Checkpoint linear layers do not match model architecture")
    model = pack_model(model, QuantizationConfig(layers))
    model.load_state_dict(load_file(str(path)), strict=True)
    return model
