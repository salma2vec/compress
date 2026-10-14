import copy
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from compress.compression.mixed_precision import MixedPrecisionConfig
from compress.compression.packed import load_packed, pack_model, save_packed
from compress.compression.quantization import (
    ActivationAwareQuantizer,
    Int8Quantizer,
    QuantizationConfig,
)


def model():
    return nn.ModuleDict({"body": nn.Linear(5, 5), "lm_head": nn.Linear(5, 3)})


@pytest.mark.parametrize("requested", ["int4", "int8", "fp16"])
def test_output_head_floor_does_not_downgrade_higher_precision(requested):
    packed = pack_model(model(), QuantizationConfig({}, requested))
    assert packed["body"].precision == requested
    assert packed["lm_head"].precision == ("int8" if requested == "int4" else requested)


@pytest.mark.parametrize("protect", [True, False])
def test_checkpoint_retains_explicit_policy_and_effective_precision(tmp_path, protect):
    original = model()
    config = QuantizationConfig({}, "int4", protect_output_head=protect)
    packed = pack_model(copy.deepcopy(original), config)
    path = tmp_path / "model.safetensors"
    save_packed(packed, path)
    with safe_open(str(path), framework="pt") as handle:
        policy = json.loads(handle.metadata()["quantization_policy"])
    assert policy["protect_output_head"] is protect
    restored = load_packed(copy.deepcopy(original), path)
    assert restored["lm_head"].precision == ("int8" if protect else "int4")
    assert restored._compress_quantization_policy == policy
    for name, tensor in packed.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_historical_unprotected_checkpoint_is_not_silently_requantized(tmp_path):
    original = model()
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}, "int4", False))
    path = tmp_path / "legacy.safetensors"
    save_file(packed.state_dict(), str(path), metadata={
        "format_version": "1", "layers": json.dumps({"body": "int4", "lm_head": "int4"})
    })
    restored = load_packed(copy.deepcopy(original), path)
    assert restored["lm_head"].precision == "int4"
    assert restored._compress_quantization_policy["protect_output_head"] is False
    for name, tensor in packed.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_contradictory_policy_is_rejected_before_model_mutation(tmp_path):
    original = model()
    modules = list(original.modules())
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}, "int4", False))
    path = tmp_path / "contradiction.safetensors"
    save_file(packed.state_dict(), str(path), metadata={
        "format_version": "1", "layers": json.dumps({"body": "int4", "lm_head": "int4"}),
        "quantization_policy": json.dumps(QuantizationConfig({}, "int4", True).to_dict()),
    })
    with pytest.raises(ValueError, match="contradict"):
        load_packed(original, path)
    assert list(original.modules()) == modules


def test_simulation_protects_tied_embedding_when_it_precedes_head():
    original = nn.ModuleDict({"embedding": nn.Embedding(7, 5), "lm_head": nn.Linear(5, 7)})
    original["lm_head"].weight = original["embedding"].weight
    expected_q, scale, zero = Int8Quantizer.quantize_tensor(original["embedding"].weight)
    expected = Int8Quantizer.dequantize_tensor(expected_q, scale, zero)
    candidate = ActivationAwareQuantizer(QuantizationConfig({}, "int4")).quantize_model(original)
    assert candidate["embedding"].weight is candidate["lm_head"].weight
    torch.testing.assert_close(candidate["lm_head"].weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("protect", [True, False])
def test_generated_precision_plan_and_json_preserve_policy(tmp_path, protect):
    config = MixedPrecisionConfig.from_importance(
        {"body": 0.1, "lm_head": 0.0}, protect_output_head=protect
    )
    assert config.layer_dtypes == {"body": "int4", "lm_head": "int8" if protect else "int4"}
    path = tmp_path / "precision.json"
    config.to_json(path)
    restored = MixedPrecisionConfig.from_json(path)
    assert restored.protect_output_head is protect
    packed = pack_model(model(), restored.to_quantization_config())
    assert packed["lm_head"].precision == ("int8" if protect else "int4")


def test_legacy_precision_json_preserves_explicit_old_assignment(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"lm_head": "int4"}))
    restored = MixedPrecisionConfig.from_json(path)
    assert restored.layer_dtypes == {"lm_head": "int4"}
    assert restored.to_quantization_config().protect_output_head is False


def test_custom_output_head_path_is_supported():
    original = nn.ModuleDict({"body": nn.Linear(5, 5), "output_projection": nn.Linear(5, 3)})
    packed = pack_model(original, QuantizationConfig(
        {}, "int4", output_head_names=("output_projection",)
    ))
    assert packed["output_projection"].precision == "int8"
    assert packed["body"].precision == "int4"
