import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import Int8Quantizer, QuantizationConfig
from compress.evaluation.latency import measure_latency
from compress.evaluation.perplexity import compute_perplexity
from compress.profiling.activation_recorder import ActivationRecorder


@pytest.mark.parametrize('value', [-5., 0., 4.])
def test_int8_constant_rows(value):
    w = torch.full((3, 12), value)
    q, s, z = Int8Quantizer.quantize_tensor(w)
    torch.testing.assert_close(Int8Quantizer.dequantize_tensor(q, s, z), w)


@pytest.mark.parametrize('precision', ['int4', 'int8', 'fp16'])
def test_packed_checkpoint_roundtrip_and_storage(precision, tmp_path):
    torch.manual_seed(1)
    original = nn.Sequential(nn.Linear(31, 33), nn.GELU(), nn.Linear(33, 17)).eval()
    packed = pack_model(copy.deepcopy(original), QuantizationConfig({}, precision))
    x = torch.randn(2, 31)
    assert tensor_bytes(packed) < tensor_bytes(original)
    assert (packed(x) - original(x)).abs().mean() < .05
    path = tmp_path / 'model.pt'
    save_packed(packed, path)
    restored = load_packed(copy.deepcopy(original), path)
    torch.testing.assert_close(restored(x), packed(x), rtol=0, atol=0)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[int(c) for c in text]])}


class CountingLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.scored = []
    def forward(self, input_ids, labels):
        self.scored.extend(labels[:, 1:][labels[:, 1:] != -100].tolist())
        return SimpleNamespace(loss=torch.tensor(math.log(2)))


def test_sliding_perplexity_scores_each_target_once_and_restores_mode():
    model = CountingLM().train()
    assert compute_perplexity(model, Tokenizer(), ['0123456789'], max_length=4, stride=2) == pytest.approx(2)
    assert model.scored == list(range(1, 10))
    assert model.training


def test_generation_failure_propagates():
    model = CountingLM()
    def broken(**kwargs):
        raise RuntimeError('generation failed')
    model.generate = broken
    with pytest.raises(RuntimeError, match='generation failed'):
        measure_latency(model, Tokenizer(), prompt='12')
    assert model.training


def test_early_generation_counts_actual_tokens():
    model = CountingLM()
    model.generate = lambda **kw: torch.cat([kw['input_ids'], torch.ones(1, 2, dtype=torch.long)], dim=1)
    result = measure_latency(model, Tokenizer(), prompt='12', n_tokens=50, n_runs=2)
    assert result['generated_tokens'] == [2, 2]
    assert result['tokens_per_sec'] == pytest.approx(4 / sum(result['latencies_s']))


def test_recorder_cleans_hooks_on_failure():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(2, 2)
        def forward(self, **kwargs):
            raise RuntimeError('failed')
    model = Broken().train()
    recorder = ActivationRecorder(model)
    with pytest.raises(RuntimeError):
        recorder.record(['12'], Tokenizer())
    assert not recorder._hooks
    assert not model.mlp._forward_hooks
    assert model.training


def test_correction_only_changes_bias_and_restores_modes():
    from compress.reconstruction.correction import PostCompressionCorrection
    teacher = nn.Linear(3, 2).train()
    student = copy.deepcopy(teacher).train()
    with torch.no_grad():
        student.bias.add_(1)
    weight = student.weight.clone()
    losses = PostCompressionCorrection(student, teacher).correct_bias([torch.ones(4, 3)], lr=.05, steps=3)
    assert losses[-1] < losses[0]
    torch.testing.assert_close(student.weight, weight)
    assert student.weight.grad is None
    assert student.weight.requires_grad and student.training and teacher.training


def test_task_accuracy_rejects_mismatched_labels():
    from compress.evaluation.task_eval import TaskEvaluator
    with pytest.raises(ValueError):
        TaskEvaluator(CountingLM(), Tokenizer()).accuracy(['12', '23'], [1])


def test_multiple_choice_masks_context_and_counts_only_completion():
    from compress.evaluation.task_eval import TaskEvaluator
    class OffsetTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            tokens = super().__call__(text)
            tokens['offset_mapping'] = torch.tensor([[[i, i + 1] for i in range(len(text))]])
            return tokens
    model = CountingLM()
    score = TaskEvaluator(model, OffsetTokenizer()).multiple_choice_accuracy(['01'], [['2', '34']], [0])
    assert score == 1
    assert model.scored == [2, 3, 4]
