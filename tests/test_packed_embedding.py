"""Packed embedding lookup, compatibility, and atomic checkpoint validation."""
import copy
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from compress.compression.packed import (
    PackedEmbedding,
    load_packed,
    pack_model,
    save_packed,
    tensor_bytes,
)
from compress.compression.quantization import Int8Quantizer, QuantizationConfig


def network():
    torch.manual_seed(731)
    return nn.Sequential(nn.Embedding(64, 16, padding_idx=0), nn.Linear(16, 4)).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lookup_matches_rowwise_reference_and_keeps_output_dtype(dtype):
    embedding = nn.Embedding(9, 5, padding_idx=2).to(dtype)
    # A padding row is not necessarily zero after callers replace weights.
    with torch.no_grad():
        embedding.weight[2].fill_(0.75)
    packed = PackedEmbedding(embedding)
    q, scale, _ = Int8Quantizer.quantize_tensor(embedding.weight)
    for indices in (torch.tensor([[2, 8, 2], [0, 3, 1]]), torch.empty(2, 0, dtype=torch.long),
                    torch.tensor(3), torch.tensor([1, 4], dtype=torch.int32)):
        expected = torch.nn.functional.embedding(indices, q.float() * scale[:, None]).to(dtype)
        torch.testing.assert_close(packed(indices), expected, rtol=0, atol=0)
        assert packed(indices).dtype == dtype
    assert packed.to(dtype=torch.float64)(torch.tensor([2])).dtype == torch.float64


def test_optional_packing_saves_actual_bytes_and_roundtrips(tmp_path):
    original = network()
    linear_only = pack_model(copy.deepcopy(original), QuantizationConfig({}))
    assert isinstance(linear_only[0], nn.Embedding)
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}), pack_embeddings=True)
    assert isinstance(packed[0], PackedEmbedding)
    expected = tensor_bytes(linear_only) - 64 * 16 * 4 + 64 * 16 + 64 * 4
    assert tensor_bytes(packed) == expected
    path = tmp_path / "packed.safetensors"
    save_packed(packed, path)
    with safe_open(path, framework="pt") as archive:
        assert archive.metadata()["format_version"] == "2"
    restored = load_packed(copy.deepcopy(original), path)
    indices = torch.tensor([[0, 3, 7], [63, 4, 4]])
    torch.testing.assert_close(restored(indices), packed(indices), rtol=0, atol=0)
    assert tensor_bytes(restored) == expected


def test_embedding_root_roundtrip_and_old_format_unchanged(tmp_path):
    original = nn.Embedding(8, 3)
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}), pack_embeddings=True)
    path = tmp_path / "embedding.safetensors"
    save_packed(packed, path)
    restored = load_packed(copy.deepcopy(original), path)
    torch.testing.assert_close(restored(torch.tensor([1, 6])), packed(torch.tensor([1, 6])), rtol=0, atol=0)
    linear_only = pack_model(network(), QuantizationConfig({}))
    path = tmp_path / "old.safetensors"
    save_packed(linear_only, path)
    with safe_open(path, framework="pt") as archive:
        assert archive.metadata()["format_version"] == "1"
        assert "embeddings" not in archive.metadata()
    assert isinstance(load_packed(network(), path)[0], nn.Embedding)


@pytest.mark.parametrize("problem", ["tied", "max_norm", "nonfinite", "precision", "subclass"])
def test_unsupported_embedding_rejected_without_partial_mutation(problem):
    model = network()
    config = QuantizationConfig({})
    if problem == "tied":
        model[1].weight = model[0].weight
    elif problem == "max_norm":
        model[0].max_norm = 1.0
    elif problem == "nonfinite":
        with torch.no_grad():
            model[0].weight[0, 0] = float("nan")
    elif problem == "precision":
        config.layer_dtypes["0"] = "int4"
    else:
        class CustomEmbedding(nn.Embedding):
            pass
        model[0] = CustomEmbedding(64, 16)
    modules = list(model.modules())
    with pytest.raises(ValueError):
        pack_model(model, config, pack_embeddings=True)
    assert list(model.modules()) == modules


@pytest.mark.parametrize("damage", ["padding", "scale", "dtype", "missing", "unknown_version", "policy"])
def test_invalid_embedding_archive_leaves_caller_unchanged(tmp_path, damage):
    model = network()
    original = copy.deepcopy(model.state_dict())
    modules = list(model.modules())
    packed = pack_model(copy.deepcopy(model), QuantizationConfig({}), pack_embeddings=True)
    path = tmp_path / "valid.safetensors"
    save_packed(packed, path)
    with safe_open(path, framework="pt") as archive:
        state = {key: archive.get_tensor(key) for key in archive.keys()}
        metadata = archive.metadata()
    if damage == "padding":
        metadata["embeddings"] = json.dumps({"0": {"precision": "int8", "padding_idx": 2}})
    elif damage == "scale":
        state["0.scale"].fill_(0)
    elif damage == "dtype":
        state["0.output_type"] = state["0.output_type"].double()
    elif damage == "missing":
        del state["0.packed_weight"]
    elif damage == "unknown_version":
        metadata["format_version"] = "3"
    else:
        policy = json.loads(metadata["quantization_policy"])
        policy["layer_dtypes"]["0"] = "int4"
        metadata["quantization_policy"] = json.dumps(policy)
    invalid = tmp_path / "invalid.safetensors"
    save_file(state, invalid, metadata=metadata)
    with pytest.raises(ValueError):
        load_packed(model, invalid)
    assert list(model.modules()) == modules
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_invalid_embedding_indices_are_not_silently_wrapped():
    packed = PackedEmbedding(nn.Embedding(4, 3))
    for indices in (torch.tensor([-1]), torch.tensor([4]), torch.tensor([1.0])):
        with pytest.raises((IndexError, RuntimeError)):
            packed(indices)
