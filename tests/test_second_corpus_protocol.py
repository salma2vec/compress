"""Unit checks for the frozen second-domain embedding validation."""

import math

import pytest

from benchmarks.embedding_int8_second_corpus import (
    EXPECTED_SCHEMA,
    canonical_document,
    quality_gate,
)


def incident():
    return {
        "incident_id": "test-0",
        "title": "Latency increased",
        "narrative": "A synthetic service slowed after deployment.",
        "logs": ["timeout while calling dependency"],
        "metrics": ["p95_latency_ms=900"],
        "recent_deploys": "",
        "config_diff": "timeout_ms: 5000",
        "true_action_id": "rollback",
        "true_evidence_indices": [],
        "root_cause_tag": "rollback",
    }


def test_canonical_document_is_input_only_and_deterministic():
    record = incident()
    assert set(record) == EXPECTED_SCHEMA
    first = canonical_document(record)
    record["true_action_id"] = "restart"
    record["true_evidence_indices"] = [[0, 1]]
    record["root_cause_tag"] = "restart"
    assert canonical_document(record) == first
    assert "rollback" not in first
    assert "(none)" in first


def _result(mean_nll, tensor_bytes=1):
    return {
        "mean_nll": mean_nll,
        "persistent_tensor_bytes": tensor_bytes,
        "predicted_tokens": 100,
        "checkpoint_tensor_roundtrip": True,
        "cpu_logits_exact_roundtrip": True,
        "shifted_loss_verified": True,
    }


def test_second_corpus_gate_is_frozen_and_requires_exact_bytes():
    results = {
        "fp32": _result(4.0),
        "linear_int8": _result(4.01),
        "linear_embedding_int8": _result(4.0 + math.log(1.05 - 1e-12), 71_062_592),
    }
    assert quality_gate(results)["practical_gate"] is True
    results["linear_embedding_int8"]["persistent_tensor_bytes"] -= 1
    assert quality_gate(results)["practical_gate"] is False
    results["linear_embedding_int8"]["persistent_tensor_bytes"] += 1
    results["linear_embedding_int8"]["mean_nll"] += 1e-8
    assert quality_gate(results)["practical_gate"] is False


def test_canonical_document_rejects_schema_drift():
    record = incident()
    record["new_label"] = "leak"
    with pytest.raises(ValueError, match="schema"):
        canonical_document(record)
