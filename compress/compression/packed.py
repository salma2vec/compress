"""Portable weight-only storage; dequantizes for ordinary PyTorch matmul.

This backend reduces stored tensor bytes, not arithmetic cost or peak memory.
It intentionally makes no low-bit kernel throughput claim.
"""
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file
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


class PackedEmbedding(nn.Module):
    """Inference-only row-scaled INT8 lookup; dequantize only requested rows.

    This is opt-in and supports ordinary untied nn.Embedding modules. Renormalizing
    max_norm embeddings and architectures accessing child .weight are unsupported.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        if type(embedding) is not nn.Embedding:
            raise ValueError("Custom embedding subclasses require an explicit adapter")
        if embedding.max_norm is not None:
            raise ValueError("Embedding max_norm mutates weights and is unsupported")
        if not embedding.weight.is_floating_point():
            raise ValueError("Embedding weights must be floating point")
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.precision = "int8"
        quantized, scale, _ = Int8Quantizer.quantize_tensor(embedding.weight.detach())
        self.register_buffer("packed_weight", quantized)
        self.register_buffer("scale", scale)
        # An empty buffer preserves output dtype across state_dict and .to(dtype).
        self.register_buffer("output_type", embedding.weight.detach().new_empty(0))

    def forward(self, indices):
        rows = torch.nn.functional.embedding(indices, self.packed_weight)
        scales = torch.nn.functional.embedding(indices, self.scale[:, None])
        return (rows.float() * scales).to(self.output_type.dtype)


def _prepare_packed_modules(model: nn.Module, config: QuantizationConfig, pack_embeddings=False):
    """Construct every replacement before changing the supplied model."""
    QuantizationConfig.from_dict(config.to_dict())
    if type(pack_embeddings) is not bool:
        raise ValueError("pack_embeddings must be boolean")
    kinds = (nn.Linear, nn.Embedding) if pack_embeddings else (nn.Linear,)
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, kinds)]
    if not modules:
        raise ValueError("No supported modules found")
    parameter_ids = [id(p) for _, p in model.named_parameters(remove_duplicate=False)]
    for name, module in modules:
        if parameter_ids.count(id(module.weight)) > 1:
            raise ValueError(f"Tied/shared weight is unsupported: {name}")
        if isinstance(module, nn.Embedding) and config.layer_dtypes.get(name, "int8") != "int8":
            raise ValueError("Packed embeddings require an explicit INT8 assignment")
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(parent, nn.MultiheadAttention):
            raise ValueError("MultiheadAttention accesses out_proj.weight directly; use fake quantization")
    return [(name, PackedEmbedding(module) if isinstance(module, nn.Embedding)
             else PackedLinear(module, config.dtype_for(name)))
            for name, module in modules]


def _install_packed_modules(model, replacements):
    for name, packed in replacements:
        if not name:
            return packed
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, packed)
    return model


def pack_model(model: nn.Module, config: QuantizationConfig, *, pack_embeddings=False):
    """Replace ordinary nn.Linear modules in place; reject tied/shared weights.

    Some architectures directly access child .weight (e.g. MultiheadAttention).
    Such modules cannot use this portable backend; the caller must exclude them.

    With pack_embeddings=True, ordinary untied nn.Embedding modules also use INT8
    storage, independently of the linear precision default. Their explicit config
    entries, if present, must be int8. Existing default packing is unchanged.
    """
    model = _install_packed_modules(model, _prepare_packed_modules(model, config, pack_embeddings))
    model._compress_quantization_policy = config.to_dict()
    return model


def tensor_bytes(model):
    """Persistent parameter + buffer bytes, excluding temporary dequantization."""
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


def save_packed(model, path):
    spec = {name: module.precision for name, module in model.named_modules() if isinstance(module, PackedLinear)}
    embeddings = {name: {"precision": module.precision, "padding_idx": module.padding_idx}
                  for name, module in model.named_modules() if isinstance(module, PackedEmbedding)}
    if not spec and not embeddings:
        raise ValueError("Model has no packed layers")
    policy = getattr(model, "_compress_quantization_policy", None)
    if policy is None:
        policy = QuantizationConfig(spec, protect_output_head=False).to_dict()
    config = QuantizationConfig.from_dict(policy)
    if any(config.dtype_for(name) != precision for name, precision in spec.items()):
        raise ValueError("Packed layers no longer match recorded quantization policy")
    if any(config.layer_dtypes.get(name, "int8") != "int8" for name in embeddings):
        raise ValueError("Packed embeddings no longer match recorded quantization policy")
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in model.state_dict().items()}
    metadata = {"format_version": "2" if embeddings else "1", "layers": json.dumps(spec),
                "quantization_policy": json.dumps(policy)}
    if embeddings:
        metadata["embeddings"] = json.dumps(embeddings)
    save_file(state, str(path), metadata=metadata)


def load_packed(model, path):
    """Restore a validated tensor archive into a trusted caller-created model.

    Checkpoint schema errors leave the caller's modules and tensors unchanged.
    This does not authenticate weights or sandbox the supplied model's Python.
    """
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        archive_version = metadata.get("format_version")
        if archive_version not in {"1", "2"}:
            raise ValueError("Unsupported packed checkpoint version")
        embeddings = json.loads(metadata.get("embeddings", "null")) if archive_version == "2" else {}
        if not isinstance(embeddings, dict) or not all(isinstance(name, str) for name in embeddings):
            raise ValueError("Checkpoint embeddings must be a name-to-spec mapping")
        if archive_version == "2" and not embeddings:
            raise ValueError("Version 2 requires packed embedding metadata")
        if archive_version == "1" and "embeddings" in metadata:
            raise ValueError("Version 1 cannot describe packed embeddings")
        layers = json.loads(metadata.get("layers", "null"))
        if not isinstance(layers, dict) or not all(isinstance(name, str) for name in layers):
            raise ValueError("Checkpoint layers must be a name-to-precision mapping")
        policy = (json.loads(metadata["quantization_policy"])
                  if "quantization_policy" in metadata else
                  QuantizationConfig(layers, protect_output_head=False).to_dict())
        config = QuantizationConfig.from_dict(policy)
        if any(config.dtype_for(name) != precision for name, precision in layers.items()):
            raise ValueError("Checkpoint layers contradict quantization policy")
        # Read metadata and tensors through one handle; do not reopen a changed path.
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    expected = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    if set(layers) != expected:
        raise ValueError("Checkpoint linear layers do not match model architecture")
    if embeddings:
        expected_embeddings = {name: module for name, module in model.named_modules()
                               if isinstance(module, nn.Embedding)}
        if set(embeddings) != set(expected_embeddings):
            raise ValueError("Checkpoint embeddings do not match model architecture")
        for name, module in expected_embeddings.items():
            entry = embeddings[name]
            if (not isinstance(entry, dict) or set(entry) != {"precision", "padding_idx"}
                    or entry["precision"] != "int8"
                    or (entry["padding_idx"] is not None and type(entry["padding_idx"]) is not int)
                    or entry["padding_idx"] != module.padding_idx
                    or config.layer_dtypes.get(name, "int8") != "int8"):
                raise ValueError("Checkpoint embedding policy differs from model architecture")
    # Restore the archive exactly, including old intentionally unprotected NF4
    # experiments. New packing defaults must never silently change old weights.
    replacements = _prepare_packed_modules(
        model, QuantizationConfig(layers, protect_output_head=False), bool(embeddings)
    )
    expected_state = dict(model.state_dict())
    for name, packed in replacements:
        prefix = f"{name}." if name else ""
        expected_state = {key: value for key, value in expected_state.items()
                          if not key.startswith(prefix)}
        expected_state.update({prefix + key: value for key, value in packed.state_dict().items()})
    if set(state) != set(expected_state):
        raise ValueError("Checkpoint tensor names do not match model architecture")
    for name, target in expected_state.items():
        source = state[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
        if (source.is_floating_point() or source.is_complex()) and not bool(torch.isfinite(source).all()):
            raise ValueError(f"Checkpoint contains nonfinite tensor: {name}")
    for name, packed in replacements:
        prefix = f"{name}." if name else ""
        if packed.precision != "fp16" and not bool((state[prefix + "scale"] > 0).all()):
            raise ValueError(f"Checkpoint contains nonpositive quantization scale: {name}")
    model = _install_packed_modules(model, replacements)
    model.load_state_dict(state, strict=True)
    model._compress_quantization_policy = policy
    return model
