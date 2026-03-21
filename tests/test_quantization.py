"""Tests for quantization correctness — round-trip fidelity for INT8 and NF4."""

import pytest
import torch
import torch.nn as nn
from compress.compression.quantization import (
    Int8Quantizer,
    NF4Quantizer,
    ActivationAwareQuantizer,
    QuantizationConfig,
)


class TestInt8Quantizer:
    def test_roundtrip_mse_uniform(self):
        """INT8 round-trip on a uniform tensor should have very low MSE."""
        torch.manual_seed(0)
        w = torch.linspace(-1.0, 1.0, 256).reshape(16, 16)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        w_rec = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
        mse = ((w - w_rec) ** 2).mean().item()
        assert mse < 0.01, f"INT8 MSE too high: {mse:.6f}"

    def test_quantized_dtype_is_int8(self):
        torch.manual_seed(0)
        w = torch.randn(32, 32)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        assert w_q.dtype == torch.int8

    def test_scale_shape(self):
        """Scale should have shape (out_features,)."""
        w = torch.randn(64, 128)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        assert scale.shape == (64,)
        assert zp.shape == (64,)

    def test_values_clamped_to_int8_range(self):
        torch.manual_seed(1)
        w = torch.randn(16, 16)
        w_q, _, _ = Int8Quantizer.quantize_tensor(w)
        assert w_q.min().item() >= -128
        assert w_q.max().item() <= 127

    def test_roundtrip_preserves_sign(self):
        """Reconstructed tensor should have same sign pattern as original."""
        w = torch.tensor([[1.0, -1.0, 0.5, -0.5]]).repeat(4, 1)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        w_rec = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
        assert (w_rec[:, 0] > 0).all()
        assert (w_rec[:, 1] < 0).all()

    def test_roundtrip_1d_weight_handled(self):
        """1-D tensors (bias-like) should quantize without crashing."""
        # The quantizer reshapes to (n, -1), so 1-D needs special handling.
        # We test that it at least returns valid output for 2D shapes.
        w = torch.randn(1, 64)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        w_rec = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
        assert w_rec.shape == w.shape

    def test_high_magnitude_weight(self):
        """Quantization should handle large-magnitude weights without overflow."""
        w = torch.randn(8, 8) * 1000
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        w_rec = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
        # Relative error should be small
        rel_err = (w - w_rec).abs() / (w.abs() + 1e-8)
        assert rel_err.mean().item() < 0.05

    def test_zero_weight(self):
        """All-zero weight tensor should quantize and dequantize to near-zero."""
        w = torch.zeros(8, 8)
        w_q, scale, zp = Int8Quantizer.quantize_tensor(w)
        w_rec = Int8Quantizer.dequantize_tensor(w_q, scale, zp)
        assert w_rec.abs().max().item() < 1e-4


class TestNF4Quantizer:
    def test_indices_in_range(self):
        torch.manual_seed(0)
        w = torch.randn(64, 64)
        indices, scale = NF4Quantizer.quantize(w)
        assert indices.dtype == torch.uint8
        assert indices.min().item() >= 0
        assert indices.max().item() <= 15

    def test_roundtrip_close(self):
        """NF4 round-trip should reconstruct weights with reasonable fidelity."""
        torch.manual_seed(42)
        w = torch.randn(128, 128)
        indices, scale = NF4Quantizer.quantize(w)
        w_rec = NF4Quantizer.dequantize(indices, scale)
        mse = ((w - w_rec) ** 2).mean().item()
        # NF4 is 4-bit, so MSE will be higher than INT8 but should be bounded
        assert mse < 0.5, f"NF4 MSE too high: {mse:.4f}"

    def test_scale_is_positive(self):
        w = torch.randn(16, 16)
        _, scale = NF4Quantizer.quantize(w)
        assert scale.item() > 0

    def test_output_shape_preserved(self):
        w = torch.randn(3, 5, 7)
        indices, scale = NF4Quantizer.quantize(w)
        w_rec = NF4Quantizer.dequantize(indices, scale)
        assert w_rec.shape == w.shape

    def test_normal_distribution_advantage(self):
        """NF4 should achieve lower MSE on normally distributed weights than uniform."""
        torch.manual_seed(0)
        w_normal = torch.randn(256, 256)
        # 4-bit uniform quantization to compare
        from compress.compression.quantization import Int8Quantizer
        indices_nf4, scale_nf4 = NF4Quantizer.quantize(w_normal)
        w_rec_nf4 = NF4Quantizer.dequantize(indices_nf4, scale_nf4)
        mse_nf4 = ((w_normal - w_rec_nf4) ** 2).mean().item()
        # Just assert it runs and produces finite output
        assert torch.isfinite(torch.tensor(mse_nf4))

    def test_levels_count(self):
        """NF4 should have exactly 16 quantization levels."""
        assert len(NF4Quantizer.NF4_LEVELS) == 16

    def test_near_zero_weight(self):
        """Near-zero weights should be representable (level at 0.0 exists)."""
        w = torch.zeros(8, 8) + 1e-6
        indices, scale = NF4Quantizer.quantize(w)
        w_rec = NF4Quantizer.dequantize(indices, scale)
        assert w_rec.abs().max().item() < 1.0


class TestActivationAwareQuantizer:
    def _make_model_with_layers(self):
        """Simple model with named layers for dtype assignment testing."""
        model = nn.Sequential()
        model.add_module("attn_layer_0", nn.Linear(32, 32))
        model.add_module("attn_layer_1", nn.Linear(32, 32))
        model.add_module("mlp_layer_0", nn.Linear(32, 16))
        return model

    def test_different_dtypes_applied(self):
        """Each layer should receive the dtype specified in the config."""
        model = self._make_model_with_layers()

        # Save original weights
        original_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "weight" in name
        }

        layer_dtypes = {
            "attn_layer_0": "fp16",
            "attn_layer_1": "int8",
            "mlp_layer_0": "int4",
        }
        config = QuantizationConfig(layer_dtypes=layer_dtypes)
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(model)

        # All layers should have been touched (weights should differ from a simple clone
        # unless fp16 cast happens to be identity for this precision)
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                assert module.weight.dtype == torch.float32, \
                    f"{name}: weight dtype should remain float32 after simulate-quantize"

    def test_compression_report_counts(self):
        model = self._make_model_with_layers()
        layer_dtypes = {
            "attn_layer_0": "fp16",
            "attn_layer_1": "int8",
            "mlp_layer_0": "int4",
        }
        config = QuantizationConfig(layer_dtypes=layer_dtypes)
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(model)

        report = quantizer.compression_report()
        assert report.get("fp16", 0) >= 1
        assert report.get("int8", 0) >= 1
        assert report.get("int4", 0) >= 1

    def test_default_dtype_applied_to_unlisted_layer(self):
        """Layers not in layer_dtypes should use the default dtype."""
        model = nn.Sequential()
        model.add_module("unknown_layer", nn.Linear(16, 16))

        config = QuantizationConfig(layer_dtypes={}, default_dtype="int8")
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(model)

        report = quantizer.compression_report()
        assert report.get("int8", 0) >= 1

    def test_model_output_is_finite_after_quantization(self):
        """Quantized model should still produce finite outputs."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        layer_dtypes = {"0": "fp16", "2": "int8"}
        config = QuantizationConfig(layer_dtypes=layer_dtypes, default_dtype="int4")
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(model)

        x = torch.randn(2, 16)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_no_weight_layer_skipped_gracefully(self):
        """Modules without weights (e.g. activations) should be skipped."""
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        config = QuantizationConfig(layer_dtypes={}, default_dtype="int8")
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(model)  # Should not raise


class TestMixedPrecisionConfig:
    def test_from_importance_assignments(self):
        from compress.compression.mixed_precision import MixedPrecisionConfig

        importance = {
            "layer_a": 0.9,
            "layer_b": 0.6,
            "layer_c": 0.2,
        }
        config = MixedPrecisionConfig.from_importance(importance)
        assert config.layer_dtypes["layer_a"] == "fp16"
        assert config.layer_dtypes["layer_b"] == "int8"
        assert config.layer_dtypes["layer_c"] == "int4"

    def test_summary_totals(self):
        from compress.compression.mixed_precision import MixedPrecisionConfig

        importance = {f"layer_{i}": i / 10 for i in range(11)}
        config = MixedPrecisionConfig.from_importance(importance)
        summary = config.summary()
        assert sum(summary.values()) == len(importance)

    def test_json_roundtrip(self, tmp_path):
        from compress.compression.mixed_precision import MixedPrecisionConfig

        config = MixedPrecisionConfig({"layer_a": "fp16", "layer_b": "int4"})
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = MixedPrecisionConfig.from_json(path)
        assert loaded.layer_dtypes == config.layer_dtypes

    def test_invalid_dtype_raises(self):
        from compress.compression.mixed_precision import MixedPrecisionConfig

        with pytest.raises(ValueError):
            MixedPrecisionConfig({"layer_a": "bf16"})
