"""Tests for activation recording and importance scoring."""

import pytest
import torch
import torch.nn as nn
from compress.profiling.activation_recorder import ActivationRecorder, ActivationProfile, LayerStats
from compress.profiling.importance import ImportanceScorer


class SimpleMLP(nn.Module):
    """Minimal 2-layer MLP for fast hook testing."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, input_ids, **kwargs):
        x = input_ids.float()
        # Flatten to (batch, features)
        x = x.view(x.shape[0], -1)
        if x.shape[1] < 32:
            x = torch.nn.functional.pad(x, (0, 32 - x.shape[1]))
        elif x.shape[1] > 32:
            x = x[:, :32]
        out = self.mlp(x)

        class Out:
            logits = out

        return Out()


class DummyTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128, **kwargs):
        torch.manual_seed(hash(text) % (2**31))
        ids = torch.randint(0, 50, (1, 8))
        return {"input_ids": ids}

    @property
    def pad_token(self):
        return "[PAD]"

    @pad_token.setter
    def pad_token(self, v):
        pass


@pytest.fixture
def simple_model():
    return SimpleMLP()


@pytest.fixture
def tokenizer():
    return DummyTokenizer()


class TestActivationRecorder:
    def test_hooks_attach_and_detach(self, simple_model):
        recorder = ActivationRecorder(simple_model)
        assert len(recorder._hooks) == 0
        recorder.attach()
        assert len(recorder._hooks) > 0
        recorder.detach()
        assert len(recorder._hooks) == 0

    def test_no_hooks_after_detach(self, simple_model):
        """Verify that hooks are cleaned up and no references remain."""
        recorder = ActivationRecorder(simple_model)
        recorder.attach()
        recorder.detach()

        # Re-attach and detach again to test idempotency
        recorder.attach()
        recorder.detach()
        assert len(recorder._hooks) == 0

    def test_profile_has_layer_names(self, simple_model, tokenizer):
        """Profile should contain at least one layer entry matching 'mlp'."""
        recorder = ActivationRecorder(simple_model)
        texts = [f"text {i}" for i in range(5)]
        profile = recorder.record(texts, tokenizer, n_batches=5)

        assert isinstance(profile, ActivationProfile)
        assert len(profile.layers) > 0
        assert any("mlp" in name for name in profile.layers)

    def test_layer_stats_are_finite(self, simple_model, tokenizer):
        """All recorded layer statistics should be finite numbers."""
        recorder = ActivationRecorder(simple_model)
        texts = [f"text {i}" for i in range(5)]
        profile = recorder.record(texts, tokenizer, n_batches=5)

        for name, stats in profile.layers.items():
            assert torch.isfinite(torch.tensor(stats.mean_magnitude)), f"{name}: mean_magnitude not finite"
            assert torch.isfinite(torch.tensor(stats.sparsity)), f"{name}: sparsity not finite"
            assert 0.0 <= stats.sparsity <= 1.0, f"{name}: sparsity out of [0, 1]"
            assert stats.n_samples > 0, f"{name}: n_samples is 0"

    def test_reset_clears_stats(self, simple_model, tokenizer):
        recorder = ActivationRecorder(simple_model)
        texts = [f"text {i}" for i in range(3)]
        recorder.record(texts, tokenizer, n_batches=3)
        assert len(recorder._stats) > 0
        recorder.reset()
        assert len(recorder._stats) == 0

    def test_profile_roundtrip(self, simple_model, tokenizer):
        """ActivationProfile should survive to_dict/from_dict roundtrip."""
        recorder = ActivationRecorder(simple_model)
        texts = [f"text {i}" for i in range(5)]
        profile = recorder.record(texts, tokenizer, n_batches=5)

        d = profile.to_dict()
        restored = ActivationProfile.from_dict(d)

        for name in profile.layers:
            orig = profile.layers[name]
            rest = restored.layers[name]
            assert abs(orig.mean_magnitude - rest.mean_magnitude) < 1e-6
            assert abs(orig.sparsity - rest.sparsity) < 1e-6

    def test_epsilon_affects_sparsity(self, simple_model, tokenizer):
        """Higher epsilon should yield higher or equal sparsity."""
        texts = [f"text {i}" for i in range(5)]

        recorder_low = ActivationRecorder(simple_model, epsilon=1e-6)
        profile_low = recorder_low.record(texts, tokenizer, n_batches=5)

        recorder_high = ActivationRecorder(simple_model, epsilon=10.0)
        profile_high = recorder_high.record(texts, tokenizer, n_batches=5)

        for name in profile_low.layers:
            if name in profile_high.layers:
                assert profile_high.layers[name].sparsity >= profile_low.layers[name].sparsity - 1e-6


class TestImportanceScorer:
    def _make_profile(self) -> ActivationProfile:
        profile = ActivationProfile()
        profile.layers["layer.0.attn"] = LayerStats(
            mean_magnitude=2.5, sparsity=0.1, variance=0.5, n_samples=10
        )
        profile.layers["layer.1.attn"] = LayerStats(
            mean_magnitude=0.01, sparsity=0.95, variance=0.001, n_samples=10
        )
        profile.layers["layer.0.mlp"] = LayerStats(
            mean_magnitude=1.0, sparsity=0.4, variance=0.2, n_samples=10
        )
        return profile

    def test_scores_in_range(self):
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        scores = scorer.layer_importance()

        for name, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{name}: score {score} out of [0, 1]"

    def test_max_score_is_one(self):
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        scores = scorer.layer_importance()

        if scores:
            assert abs(max(scores.values()) - 1.0) < 1e-6

    def test_important_layer_scores_higher(self):
        """Layer with high magnitude and low sparsity should outscore dormant layer."""
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        scores = scorer.layer_importance()

        # layer.0.attn has high magnitude and low sparsity, should beat layer.1.attn
        assert scores["layer.0.attn"] > scores["layer.1.attn"]

    def test_head_importance_subset(self):
        """head_importance should only return attention layers."""
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        head_scores = scorer.head_importance()

        for name in head_scores:
            assert "attn" in name or "attention" in name

    def test_recommended_dtype_valid(self):
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        valid_dtypes = {"fp16", "int8", "int4"}

        for name in profile.layers:
            dtype = scorer.recommended_dtype(name)
            assert dtype in valid_dtypes, f"Invalid dtype '{dtype}' for layer '{name}'"

    def test_recommended_dtype_high_importance_is_fp16(self):
        profile = ActivationProfile()
        # Very high importance: high magnitude, zero sparsity
        profile.layers["important.layer"] = LayerStats(
            mean_magnitude=100.0, sparsity=0.0, variance=1.0, n_samples=10
        )
        # Very low importance: low magnitude, high sparsity
        profile.layers["dormant.layer"] = LayerStats(
            mean_magnitude=0.0001, sparsity=0.99, variance=0.0, n_samples=10
        )
        scorer = ImportanceScorer(profile)

        assert scorer.recommended_dtype("important.layer") == "fp16"
        assert scorer.recommended_dtype("dormant.layer") == "int4"

    def test_summary_keys(self):
        profile = self._make_profile()
        scorer = ImportanceScorer(profile)
        summary = scorer.summary()

        assert "n_layers" in summary
        assert "mean_importance" in summary
        assert "dtype_breakdown" in summary
        assert summary["n_layers"] == len(profile.layers)

    def test_empty_profile(self):
        profile = ActivationProfile()
        scorer = ImportanceScorer(profile)
        scores = scorer.layer_importance()
        assert scores == {}
        summary = scorer.summary()
        assert summary["n_layers"] == 0
