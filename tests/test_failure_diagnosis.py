import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from benchmarks import failure_diagnosis as diagnosis
from compress.compression.packed import PackedLinear

CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def scalar_reference(weights, precision):
    """Deliberately independent scalar nearest-codebook calculation."""
    output = []
    for row in weights.tolist():
        maximum = max(abs(value) for value in row)
        if precision == "int4":
            scale = max(maximum, torch.finfo(torch.float32).tiny)
            output.append([
                min(CODEBOOK, key=lambda level: abs(value / scale - level)) * scale
                for value in row
            ])
        else:
            scale = max(maximum / 127, torch.finfo(torch.float32).tiny)
            output.append([max(-127, min(127, round(value / scale))) * scale for value in row])
    return torch.tensor(output, dtype=torch.float32)


@pytest.mark.parametrize("precision", ["int8", "int4"])
@pytest.mark.parametrize(
    "values",
    [
        [[0.0] * 5, [2.0] * 5, [-3.0] * 5],
        [[-1.0, -0.7, -0.1, 0.2, 0.8], [0.31, -0.25, 0.7, -0.9, 1.0]],
        [[1e6, 0.1, -0.3, 2.0, -7.0, 0.0, 0.2]],
    ],
)
def test_packed_quantizers_match_scalar_oracle(precision, values):
    weights = torch.tensor(values, dtype=torch.float32)
    linear = nn.Linear(weights.shape[1], weights.shape[0], bias=False)
    linear.weight.data.copy_(weights)
    packed = PackedLinear(linear, precision)
    torch.testing.assert_close(packed.dequantize(), scalar_reference(weights, precision))
    if precision == "int4":
        assert packed.packed_weight.numel() == (weights.numel() + 1) // 2


def tiny_model():
    return nn.ModuleDict({"body": nn.Linear(5, 5), "lm_head": nn.Linear(5, 3, bias=False)})


def test_head_only_intervention_preserves_float_body_exactly():
    model = tiny_model()
    before = diagnosis.state_hashes(model)
    plans = diagnosis.intervention_plans(model)
    assert list(plans) == list(diagnosis.METHODS)
    assert plans["fp32"] == {}
    assert plans["fp32_body_nf4_head"] == {"lm_head": "int4"}
    candidate = diagnosis.apply_intervention(copy.deepcopy(model), plans["fp32_body_nf4_head"])
    assert isinstance(candidate["body"], nn.Linear)
    assert candidate["body"].weight.dtype == torch.float32
    torch.testing.assert_close(candidate["body"].weight, model["body"].weight, rtol=0, atol=0)
    assert isinstance(candidate["lm_head"], PackedLinear)
    assert diagnosis.state_hashes(model) == before


def test_partial_checkpoint_roundtrip_and_changed_contents_fail(tmp_path):
    model = tiny_model()
    plan = {"lm_head": "int4"}
    candidate = diagnosis.apply_intervention(copy.deepcopy(model), plan)
    hashes = diagnosis.state_hashes(candidate)
    path = tmp_path / "model.safetensors"
    save_file(candidate.state_dict(), str(path))
    restored = diagnosis.restore_checkpoint(path, tiny_model, plan, hashes)
    assert diagnosis.state_hashes(restored) == hashes
    assert isinstance(restored["body"], nn.Linear)
    x = torch.randn(2, 5)
    torch.testing.assert_close(restored["lm_head"](x), candidate["lm_head"](x), rtol=0, atol=0)
    changed = candidate.state_dict()
    changed["body.weight"] = changed["body.weight"] + 1
    save_file(changed, str(path))
    with pytest.raises(ValueError, match="changed tensor"):
        diagnosis.restore_checkpoint(path, tiny_model, plan, hashes)


def measurements(losses):
    return [
        {"method": name, "predicted_tokens": 100, "negative_log_likelihood": loss * 100}
        for name, loss in zip(diagnosis.METHODS, losses, strict=True)
    ]


@pytest.mark.parametrize(
    ("losses", "status"),
    [
        ([4, 4.1, 14, 5, 13], "supported"),
        ([4, 4.1, 14, 7, 13], "rejected"),
        ([4, 4.1, 14, 5, 10], "rejected"),
        ([4, 4.1, 5, 4.2, 4.2], "not_reproduced"),
        ([4, 4.1, 4, 4, 4], "not_reproduced"),
    ],
)
def test_diagnosis_requires_both_fixed_fractions_and_failure_reproduction(losses, status):
    assert diagnosis.diagnosis_gate(measurements(losses))["status"] == status


def test_diagnosis_preserves_interactions_instead_of_clipping():
    result = diagnosis.diagnosis_gate(measurements([4, 4.1, 14, 3, 16]))
    assert result["head_rescue_fraction"] == pytest.approx(1.1)
    assert result["head_only_fraction"] == pytest.approx(1.2)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "tokens", "fractional", "nan"])
def test_diagnosis_rejects_incomplete_or_incomparable_results(failure):
    rows = measurements([4, 4.1, 14, 5, 13])
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "tokens":
        rows[-1]["predicted_tokens"] = 101
    elif failure == "fractional":
        rows[-1]["predicted_tokens"] = 100.0
    else:
        rows[-1]["negative_log_likelihood"] = float("nan")
    with pytest.raises(ValueError):
        diagnosis.diagnosis_gate(rows)


def test_asset_validation_fails_before_model_loading_and_retains_failure(tmp_path, monkeypatch):
    pilot = {
        "model_revision": diagnosis.MODEL_REVISION,
        "dataset_revision": diagnosis.DATASET_REVISION,
        "asset_sha256": {name: "not-present" for name in diagnosis.MODEL_FILES},
    }
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot))
    monkeypatch.setattr(diagnosis, "source_provenance", lambda: {"source_clean": True})
    model_calls = []
    monkeypatch.setattr(
        diagnosis.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model_calls.append(kw)
    )
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Missing or changed"):
        diagnosis.run(tmp_path / "missing", tmp_path / "missing.parquet", pilot_path, output)
    assert model_calls == []
    report = json.loads((output / "report.json").read_text())
    assert report["complete"] is False
    assert report["results"] == []
    assert report["error"]["type"] == "ValueError"
    with pytest.raises(FileExistsError):
        diagnosis.run(tmp_path / "missing", tmp_path / "missing.parquet", pilot_path, output)
