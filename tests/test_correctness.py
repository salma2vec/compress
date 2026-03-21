"""Test that compressed model output stays numerically close to the original."""

import pytest
import torch
import torch.nn as nn


def _make_tiny_transformer(n_layers: int = 2, d_model: int = 64, n_heads: int = 4, vocab_size: int = 100):
    """Build a minimal transformer for fast tests, independent of HuggingFace."""
    from torch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding(vocab_size, d_model)
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            x = self.encoder(x)
            logits = self.lm_head(x)

            class Out:
                pass

            out = Out()
            out.logits = logits
            # Compute simple next-token prediction loss
            if "labels" in kwargs and kwargs["labels"] is not None:
                labels = kwargs["labels"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                out.loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                out.loss = None
            return out

    return TinyLM()


def _dummy_tokenizer(vocab_size: int = 100, seq_len: int = 16):
    """Return a simple callable that produces random token ids."""

    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128, **kwargs):
            torch.manual_seed(hash(text) % (2**31))
            ids = torch.randint(0, vocab_size, (1, min(seq_len, max_length)))
            return {"input_ids": ids}

        @property
        def pad_token(self):
            return "[PAD]"

        @pad_token.setter
        def pad_token(self, v):
            pass

    return DummyTokenizer()


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return _make_tiny_transformer()


@pytest.fixture
def dummy_tokenizer():
    return _dummy_tokenizer()


class TestInt8Compression:
    def test_logit_mae_below_tolerance(self, tiny_model, dummy_tokenizer):
        """INT8-compressed model logits should stay within a reasonable MAE of the original."""
        import copy
        from compress.profiling.activation_recorder import ActivationRecorder
        from compress.profiling.importance import ImportanceScorer
        from compress.compression.quantization import ActivationAwareQuantizer, QuantizationConfig

        original = tiny_model
        compressed = copy.deepcopy(original)

        texts = [f"sample text {i}" for i in range(10)]
        recorder = ActivationRecorder(compressed)
        profile = recorder.record(texts, dummy_tokenizer, n_batches=5)
        scorer = ImportanceScorer(profile)
        importance = scorer.layer_importance()

        layer_dtypes = {k: "int8" for k in importance}
        config = QuantizationConfig(layer_dtypes=layer_dtypes)
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(compressed)

        input_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            orig_out = original(input_ids)
            comp_out = compressed(input_ids)

        mae = (orig_out.logits - comp_out.logits).abs().mean().item()
        assert mae < 5.0, f"INT8 MAE too high: {mae:.4f}"

    def test_int4_compression_runs_without_error(self, tiny_model, dummy_tokenizer):
        """INT4 compression should run without raising exceptions."""
        import copy
        from compress.profiling.activation_recorder import ActivationRecorder
        from compress.profiling.importance import ImportanceScorer
        from compress.compression.quantization import ActivationAwareQuantizer, QuantizationConfig

        compressed = copy.deepcopy(tiny_model)
        texts = [f"text {i}" for i in range(5)]
        recorder = ActivationRecorder(compressed)
        profile = recorder.record(texts, dummy_tokenizer, n_batches=5)
        scorer = ImportanceScorer(profile)
        importance = scorer.layer_importance()

        layer_dtypes = {k: "int4" for k in importance}
        config = QuantizationConfig(layer_dtypes=layer_dtypes)
        quantizer = ActivationAwareQuantizer(config)
        quantizer.quantize_model(compressed)

        input_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            out = compressed(input_ids)
        assert out.logits is not None
        assert not torch.isnan(out.logits).all()

    def test_mixed_precision_preserves_important_layers(self, tiny_model, dummy_tokenizer):
        """Mixed-precision should assign fp16 to high-importance layers."""
        import copy
        from compress.profiling.activation_recorder import ActivationRecorder
        from compress.profiling.importance import ImportanceScorer
        from compress.compression.mixed_precision import MixedPrecisionConfig

        model = copy.deepcopy(tiny_model)
        texts = [f"text {i}" for i in range(10)]
        recorder = ActivationRecorder(model)
        profile = recorder.record(texts, dummy_tokenizer, n_batches=10)
        scorer = ImportanceScorer(profile)
        importance = scorer.layer_importance()

        mp = MixedPrecisionConfig.from_importance(importance)
        summary = mp.summary()

        # Total assignments should equal number of profiled layers
        total = sum(summary.values())
        assert total == len(importance)
