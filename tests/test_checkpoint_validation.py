"""Untrusted tensor archives must fail before mutating the caller's model."""
import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from compress.compression.packed import load_packed, pack_model, save_packed
from compress.compression.quantization import QuantizationConfig


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nan", "negative_scale"])
def test_invalid_archive_preserves_modules_and_all_tensors(tmp_path, damage):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    original_modules = list(model.modules())
    original_state = copy.deepcopy(model.state_dict())
    packed = pack_model(copy.deepcopy(model), QuantizationConfig({}, "int8"))
    state = {name: value.clone() for name, value in packed.state_dict().items()}
    # This early tensor used to be copied before a later schema failure was raised.
    state["0.bias"].fill_(777)
    if damage == "missing":
        del state["1.scale"]
    elif damage == "extra":
        state["unexpected"] = torch.ones(1)
    elif damage == "shape":
        state["1.packed_weight"] = torch.ones(3, 2, dtype=torch.int8)
    elif damage == "dtype":
        state["1.packed_weight"] = state["1.packed_weight"].float()
    elif damage == "nan":
        state["1.scale"].fill_(float("nan"))
    else:
        state["1.scale"].fill_(-1)
    path = tmp_path / "invalid.safetensors"
    save_file(state, str(path), metadata={"format_version": "1", "layers": json.dumps({"0": "int8", "1": "int8"})})
    with pytest.raises(ValueError, match="Checkpoint"):
        load_packed(model, path)
    assert list(model.modules()) == original_modules
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["fp16", "int8", "int4"])
def test_root_linear_archive_roundtrip(tmp_path, precision):
    original = nn.Linear(3, 5, bias=False)
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}, precision))
    path = tmp_path / "root.safetensors"
    save_packed(packed, path)
    restored = load_packed(copy.deepcopy(original), path)
    x = torch.randn(4, 3)
    torch.testing.assert_close(restored(x), packed(x), rtol=0, atol=0)


def test_failed_pack_does_not_replace_earlier_layers():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    original_modules = list(model.modules())
    with torch.no_grad():
        model[1].weight.fill_(float("nan"))
    with pytest.raises(ValueError):
        pack_model(model, QuantizationConfig({}, "int8"))
    assert list(model.modules()) == original_modules
