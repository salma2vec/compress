import copy

from torch import nn

from benchmarks.pretrained import precision_plans, select_paragraphs
from compress.compression.packed import pack_model, tensor_bytes
from compress.compression.quantization import QuantizationConfig


def test_equal_storage_controls_preserve_weight_and_bias_budgets():
    model = nn.ModuleDict({f"layer{i}": nn.Linear(5, 3, bias=i < 4) for i in range(8)})
    scores = {f"layer{i}": float(i) for i in range(8)}
    plans = precision_plans(model, scores)
    assert plans == precision_plans(model, scores)
    assert plans["ranked_mixed"]["layer3"] == "int8"
    assert plans["ranked_mixed"]["layer0"] == "int4"
    mixed_bytes = tensor_bytes(pack_model(copy.deepcopy(model), QuantizationConfig(plans["ranked_mixed"])))
    assert any(plan != plans["ranked_mixed"] for name, plan in plans.items() if name.startswith("shuffled"))
    for name, plan in plans.items():
        if name.startswith("shuffled"):
            assert tensor_bytes(pack_model(copy.deepcopy(model), QuantizationConfig(plan))) == mixed_bytes


def test_paragraph_selection_is_seeded_and_excludes_headings_duplicates_and_short_text():
    def tokenizer(text, **kwargs):
        return {"input_ids": list(range(len(text)))}
    rows = ["= " + "x" * 128, "short", "a" * 128, "a" * 128, "b" * 129, "c" * 512, "d" * 513]
    selected = select_paragraphs(rows, tokenizer, 3, 42)
    assert {i for i, _, _ in selected} == {2, 4, 5}
    assert selected == select_paragraphs(rows, tokenizer, 3, 42)
