import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.article_holdout import article_ids, select_article_holdout
from benchmarks.sensitivity_allocation import PROTOCOL, quality_gate
from compress.compression.packed import PackedLinear, pack_model, tensor_bytes
from compress.compression.sensitivity import equal_shape_plan, module_sensitivity, shifted_loss


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(9, 4)
        self.body = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.lm_head = nn.Linear(4, 9)
        self.fail = False

    def forward(self, input_ids):
        if self.fail:
            raise RuntimeError("controlled inference failure")
        hidden = self.embed(input_ids)
        for module in self.body:
            hidden = torch.tanh(module(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_sensitivity_matches_isolated_manual_score_and_restores_model():
    torch.manual_seed(14)
    model = Tiny().train()
    before = copy.deepcopy(model.state_dict())
    inputs = [torch.tensor([[1, 2, 3, 4]])]
    scores = module_sensitivity(model, inputs)
    assert set(scores) == {f"body.{i}" for i in range(4)}
    original = model.body[0]
    losses = {}
    for precision in ("int8", "int4"):
        model.body[0] = PackedLinear(original, precision)
        losses[precision] = shifted_loss(model, inputs[0])[0] / 3
    model.body[0] = original
    assert scores["body.0"]["nll_benefit"] == pytest.approx(losses["int4"] - losses["int8"])
    assert model.training
    assert not any(isinstance(m, PackedLinear) for m in model.modules())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_inference_failure_restores_original_modules_and_training_mode():
    model = Tiny().train()
    modules = dict(model.named_modules())
    model.fail = True
    with pytest.raises(RuntimeError, match="controlled"):
        module_sensitivity(model, [torch.tensor([[1, 2]])])
    assert model.training
    assert all(model.get_submodule(name) is module for name, module in modules.items())


@pytest.mark.parametrize("fail", [False, True])
def test_mixed_training_modes_survive_success_and_failure(fail):
    model = Tiny().train()
    model.body[1].eval()
    flags = {name: module.training for name, module in model.named_modules()}
    model.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="controlled"):
            module_sensitivity(model, [torch.tensor([[1, 2, 3]])])
    else:
        module_sensitivity(model, [torch.tensor([[1, 2, 3]])])
    assert {name: module.training for name, module in model.named_modules()} == flags


def test_shape_matched_plans_have_exact_bytes_and_protected_head():
    model = Tiny()
    scores = {f"body.{i}": float(i) for i in range(4)}
    first = equal_shape_plan(model, scores)
    reverse = equal_shape_plan(model, {name: -score for name, score in scores.items()})
    assert first.layer_dtypes["lm_head"] == "int8"
    assert first.layer_dtypes != reverse.layer_dtypes
    sizes = [tensor_bytes(pack_model(copy.deepcopy(model), plan.to_quantization_config()))
             for plan in (first, reverse)]
    assert sizes[0] == sizes[1]
    with pytest.raises(ValueError, match="nonfinite"):
        equal_shape_plan(model, {**scores, "body.0": float("nan")})


def tokenizer(text, **kwargs):
    return {"input_ids": list(range(128))}


def test_article_groups_include_subheadings_and_all_exposed_article_paragraphs():
    rows = ["preamble", " = First = \n", "old", " = = Sub = = \n", "also old",
            " = Second = \n", "new1", "new2", " = Third = \n", "new3", "new4"]
    assert article_ids(rows) == [None, 1, 1, 1, 1, 5, 5, 5, 8, 8, 8]
    held = select_article_holdout(rows, tokenizer, [2], [], article_count=2)
    assert held["excluded_articles"] == [1]
    assert {row["row"] for row in held["paragraphs"]} == {6, 7, 9, 10}
    assert held == select_article_holdout(rows, tokenizer, [2], [], article_count=2)


def test_no_silent_relaxation_or_exact_calibration_overlap():
    rows = [" = First = ", "calibration", "new", " = Second = ", "a", "b"]
    held = select_article_holdout(rows, tokenizer, [], ["calibration"], article_count=1)
    assert {row["row"] for row in held["paragraphs"]} == {4, 5}
    with pytest.raises(ValueError, match="Insufficient"):
        select_article_holdout(rows, tokenizer, [], ["calibration"], article_count=2)
    with pytest.raises(ValueError, match="index"):
        select_article_holdout(rows, tokenizer, [999], [], article_count=1)


def test_extra_spaces_do_not_turn_subheading_into_new_article():
    assert article_ids([" = Main = ", " =   = Child =   = ", "paragraph"]) == [0, 0, 0]


def scored_seeds(sensitivity=2.01, magnitude=2.03):
    return [{"seed": seed, "methods": {
        name: {"mean_nll": loss, "predicted_tokens": 100, "persistent_tensor_bytes": 1000}
        for name, loss in {"sensitivity": sensitivity, "magnitude": magnitude,
                           "shuffled_11": 2.02, "shuffled_23": 2.04,
                           "shuffled_37": 2.06}.items()
    }} for seed in PROTOCOL["calibration_seeds"]]


def test_score_and_practical_gates_are_distinct():
    result = quality_gate(scored_seeds(), 2.0)
    assert result["score_gate"] and result["practical_gate"]
    # The comparison score can improve while absolute quality remains poor.
    result = quality_gate(scored_seeds(), 1.5)
    assert result["score_gate"] and not result["practical_gate"]
    result = quality_gate(scored_seeds(sensitivity=2.04), 2.0)
    assert not result["score_gate"] and result["practical_gate"]


def test_worse_seed_is_not_dropped_and_byte_or_target_mismatch_fails():
    records = scored_seeds()
    records[-1]["methods"]["sensitivity"]["mean_nll"] = 2.06
    assert not quality_gate(records, 2.0)["score_gate"]
    with pytest.raises(ValueError, match="all five"):
        quality_gate(records[:-1], 2.0)
    records = scored_seeds()
    records[0]["methods"]["magnitude"]["persistent_tensor_bytes"] += 1
    with pytest.raises(ValueError, match="identical bytes"):
        quality_gate(records, 2.0)
    records = scored_seeds()
    records[0]["methods"]["magnitude"]["predicted_tokens"] -= 1
    with pytest.raises(ValueError, match="scored targets"):
        quality_gate(records, 2.0)
