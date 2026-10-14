"""Supply-chain checks for Hugging Face model and dataset loading."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compress.huggingface_security import GPT2_REVISION, secure_pretrained_kwargs
from compress.profiling.calibration_data import build_calibration_set


def test_remote_model_requires_immutable_commit():
    with pytest.raises(ValueError, match="40-character commit SHA"):
        secure_pretrained_kwargs("organization/model", None)
    with pytest.raises(ValueError, match="40-character commit SHA"):
        secure_pretrained_kwargs("organization/model", "main")
    assert secure_pretrained_kwargs("organization/model", GPT2_REVISION) == {
        "revision": GPT2_REVISION
    }


def test_local_model_is_forced_offline_and_ignores_hub_revision(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    assert secure_pretrained_kwargs(str(model), "main") == {
        "local_files_only": True,
        "revision": None,
    }


def test_calibration_dataset_uses_immutable_revision(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"text": "long enough synthetic replacement row"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    assert build_calibration_set(n=1, min_length=0) == ["long enough synthetic replacement row"]
    assert calls == [
        (
            ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            {
                "split": "train",
                "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            },
        )
    ]
