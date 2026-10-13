import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class QuantizationConfig:
    """Per-layer quantization dtype assignments.

    Args:
        layer_dtypes: Mapping from layer name to dtype string.
            Valid values: "fp16", "int8", "int4".
        default_dtype: Dtype to apply to layers not found in layer_dtypes.
    """
    layer_dtypes: dict[str, str]
    default_dtype: str = "int8"

    def __post_init__(self):
        if any(value not in {"fp16", "int8", "int4"} for value in [self.default_dtype, *self.layer_dtypes.values()]):
            raise ValueError("Dtypes must be fp16, int8 or int4")


class Int8Quantizer:
    """Per-channel symmetric INT8 quantization.

    Scales are computed per output channel so each row of the weight matrix
    uses its own scale and zero-point. This reduces quantization error compared
    to per-tensor quantization, particularly for layers with high inter-channel
    variance.
    """

    @staticmethod
    def quantize_tensor(
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a weight tensor to INT8 using per-channel scale/zero-point.

        Args:
            w: Float weight tensor of shape (out_features, ...).

        Returns:
            Tuple of (quantized_weights, scale, zero_point).
            quantized_weights has dtype torch.int8.
            scale and zero_point have shape (out_features,).
        """
        w_flat = w.reshape(w.shape[0], -1).float()
        if not torch.isfinite(w_flat).all():
            raise ValueError("Weights must be finite")
        scale = (w_flat.abs().amax(dim=1) / 127.0).clamp(min=torch.finfo(torch.float32).tiny)
        zero_point = torch.zeros(w.shape[0], device=w.device, dtype=torch.int8)
        w_q = (w_flat / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

        return w_q.reshape(w.shape), scale, zero_point

    @staticmethod
    def dequantize_tensor(
        w_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a float tensor from INT8 quantized weights.

        Args:
            w_q: INT8 quantized weight tensor.
            scale: Per-channel scale, shape (out_features,).
            zero_point: Per-channel zero-point, shape (out_features,).

        Returns:
            Reconstructed float32 tensor.
        """
        w_flat = w_q.float().reshape(w_q.shape[0], -1)
        zp = zero_point.float().unsqueeze(1)
        sc = scale.unsqueeze(1)
        return ((w_flat - zp) * sc).reshape(w_q.shape)


class NF4Quantizer:
    """4-bit NormalFloat (NF4) quantization.

    NF4 is an information-theoretically optimal quantization for normally
    distributed weights (Dettmers et al., QLoRA, 2023). The 16 quantization
    levels are placed at the quantiles of the standard normal distribution,
    maximizing information density for typical neural network weight distributions.
    """

    # 16 quantization levels at the quantiles of N(0, 1)
    NF4_LEVELS = torch.tensor([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ])

    @classmethod
    def quantize(cls, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a weight tensor to 4-bit NF4.

        Weights are normalized to [-1, 1] by the absolute maximum, then mapped
        to the nearest NF4 quantization level.

        Args:
            w: Float weight tensor.

        Returns:
            Tuple of (indices, scale).
            indices is uint8 tensor with values in [0, 15].
            scale is a scalar tensor.
        """
        if not w.numel() or not torch.isfinite(w).all():
            raise ValueError("Weights must be nonempty and finite")
        scale = w.float().abs().max()
        w_norm = (w / scale.clamp(min=1e-8)).clamp(-1, 1)
        levels = cls.NF4_LEVELS.to(w.device)
        boundaries = (levels[:-1] + levels[1:]) / 2
        indices = torch.bucketize(w_norm.contiguous(), boundaries).to(torch.uint8)
        return indices, scale

    @classmethod
    def dequantize(cls, indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Reconstruct float weights from NF4 indices.

        Args:
            indices: uint8 tensor of NF4 level indices.
            scale: Scalar scale factor.

        Returns:
            Reconstructed float32 tensor.
        """
        levels = cls.NF4_LEVELS.to(indices.device)
        return levels[indices.long()] * scale


class ActivationAwareQuantizer:
    """Applies mixed-precision quantization guided by per-layer importance scores.

    Layers identified as important (high activation magnitude, low sparsity) are
    quantized to higher precision (FP16 or INT8), while dormant layers receive
    more aggressive INT4 compression.

    Args:
        config: QuantizationConfig mapping layer names to target dtypes.
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._compression_log: dict[str, str] = {}

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply per-layer quantization in-place based on the importance config.

        For each layer with a weight parameter:
          - "fp16": cast to half precision and back (simulates FP16 rounding)
          - "int8": per-channel INT8 quantize then dequantize
          - "int4": NF4 quantize then dequantize

        Args:
            model: The model to compress. Modified in-place.

        Returns:
            The same model with compressed weights.
        """
        self._compression_log.clear()
        seen = set()
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if id(module.weight) in seen or module.weight.ndim < 2:
                continue
            seen.add(id(module.weight))
            w = module.weight.detach()
            original_dtype = w.dtype
            dtype = self.config.layer_dtypes.get(name, self.config.default_dtype)
            self._compression_log[name] = dtype

            if dtype == "fp16":
                module.weight.data = w.half().float()
            elif dtype == "int8":
                if w.dim() >= 2:
                    w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
                    module.weight.data = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
                else:
                    # 1-D weight (rare, e.g. LayerNorm) — use per-tensor quantization
                    module.weight.data = w.half().float()
            elif dtype == "int4":
                indices, scale = NF4Quantizer.quantize(w)
                module.weight.data = NF4Quantizer.dequantize(indices, scale)
            module.weight.data = module.weight.data.to(original_dtype)

        return model

    def compression_report(self) -> dict[str, int]:
        """Return a count of layers compressed to each dtype."""
        counts: dict[str, int] = {}
        for dtype in self._compression_log.values():
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts
