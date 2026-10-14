"""Frozen second-domain validation for portable rowwise-INT8 embeddings."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.failure_diagnosis import file_hash, state_hashes
from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import QuantizationConfig
from compress.evaluation.perplexity import _negative_log_likelihood

PROTOCOL = {
    "version": 1,
    "corpus": "patchy synthetic incident test split",
    "generator_seed": 0,
    "split_sizes": {"train": 200, "dev": 50, "test": 50},
    "documents": 50,
    "evaluation_context": 256,
    "evaluation_stride": 128,
    "maximum_fp32_ppl_ratio": 1.05,
    "expected_candidate_tensor_bytes": 71_062_592,
    "method_order": ["fp32", "linear_int8", "linear_embedding_int8"],
    "device": "cpu",
    "threads": 1,
}

INPUT_FIELDS = ("title", "narrative", "logs", "metrics", "recent_deploys", "config_diff")
EXPECTED_SCHEMA = {
    "incident_id",
    *INPUT_FIELDS,
    "true_action_id",
    "true_evidence_indices",
    "root_cause_tag",
}
EXPECTED_SPLIT_SHA256 = {
    "train": "3014df001b1f289e8de40fda5e3dabd905476df47d963799bfc027707a165ae6",
    "dev": "5ae6462f637ad769e3c6a501c5b51107b94bba59bc1bd85982eddc220b04b7c4",
    "test": "50d447a3a4d00ea3996b889ec3e3a2e1ed2807a6bb9a786eb0010d1d492ca70e",
}
EXPECTED_GENERATOR_COMMIT = "3813eae295e9dd9b411fedba10df0791ee66e3a8"
EXPECTED_GENERATOR_SHA256 = "c46f726b99ad59096800f7e5911c60e2ffd2d17068a49983b2f5d48884ec82d5"
EXPECTED_LICENSE_SHA256 = "06dcddbb6908a0c6dd4a9e8ec822eea41d5a460a53089fecccc8a68049e99241"
EXPECTED_PREEXPOSURE_HEAD = "4db95a451e05a5a80776ba58891b7306144157ba"
EXPECTED_MODEL_ASSETS = {
    "config.json": "002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43",
    "model.safetensors": "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15",
    "special_tokens_map.json": "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad",
    "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    "tokenizer_config.json": "70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f",
}


def provenance():
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError("Freeze and commit source before preparing or running this study")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "source_commit": revision,
        "source_clean": True,
        "harness_sha256": file_hash(__file__),
        "protocol_sha256": file_hash(root / "docs/embedding-int8-second-corpus-protocol.md"),
    }


def canonical_document(record):
    """Render only incident inputs; labels are intentionally excluded."""
    if set(record) != EXPECTED_SCHEMA:
        raise ValueError("Synthetic incident record schema changed")
    if not all(isinstance(record[name], str) for name in ("incident_id", "title", "narrative")):
        raise ValueError("Incident identifiers and prose fields must be strings")
    if not all(isinstance(record[name], list) for name in ("logs", "metrics")):
        raise ValueError("Incident logs and metrics must be lists")
    if not all(isinstance(item, str) for name in ("logs", "metrics") for item in record[name]):
        raise ValueError("Incident logs and metrics must contain strings")
    if not all(isinstance(record[name], str) for name in ("recent_deploys", "config_diff")):
        raise ValueError("Deployment and configuration fields must be strings")
    return "\n".join(
        [
            f"Title: {record['title']}",
            f"Narrative: {record['narrative']}",
            "Logs:",
            *(f"- {item}" for item in record["logs"]),
            "Metrics:",
            *(f"- {item}" for item in record["metrics"]),
            f"Recent deploys: {record['recent_deploys'] or '(none)'}",
            f"Config diff: {record['config_diff'] or '(none)'}",
        ]
    )


def _load_corpus(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(records) != PROTOCOL["documents"]:
        raise ValueError("Require the complete 50-record synthetic test split")
    identifiers = [record.get("incident_id") for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Incident identifiers must be unique")
    documents = [canonical_document(record) for record in records]
    hashes = [hashlib.sha256(document.encode()).hexdigest() for document in documents]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Canonical incident documents must be unique")
    return records, documents, hashes


def _validate_frozen_assets(model_dir, corpus_file, generator_source, license_file):
    assets = {name: file_hash(Path(model_dir) / name) for name in EXPECTED_MODEL_ASSETS}
    if assets != EXPECTED_MODEL_ASSETS:
        raise ValueError("Cached model or tokenizer assets changed")
    if file_hash(corpus_file) != EXPECTED_SPLIT_SHA256["test"]:
        raise ValueError("Synthetic test corpus changed")
    if file_hash(generator_source) != EXPECTED_GENERATOR_SHA256:
        raise ValueError("Synthetic corpus generator changed")
    if file_hash(license_file) != EXPECTED_LICENSE_SHA256:
        raise ValueError("Synthetic corpus license changed")
    generator_root = Path(generator_source).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=generator_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if revision != EXPECTED_GENERATOR_COMMIT:
        raise ValueError("Synthetic corpus generator revision changed")
    return assets


def _validate_reproduction(corpus_file, reproduced_dir):
    original_dir = Path(corpus_file).parent
    reproduced_dir = Path(reproduced_dir)
    hashes = {}
    for split, expected in EXPECTED_SPLIT_SHA256.items():
        original = original_dir / f"{split}.jsonl"
        reproduced = reproduced_dir / f"{split}.jsonl"
        if file_hash(original) != expected or file_hash(reproduced) != expected:
            raise ValueError(f"Synthetic {split} split did not reproduce exactly")
        hashes[split] = expected
    return hashes


def _wikitext_overlap(document_hashes, wikitext_train, wikitext_test):
    row_hashes = set()
    rows_compared = 0
    for path in (wikitext_train, wikitext_test):
        rows = parquet.read_table(path)["text"].to_pylist()
        rows_compared += len(rows)
        row_hashes.update(hashlib.sha256(text.encode()).hexdigest() for text in rows)
    overlap = set(document_hashes) & row_hashes
    if overlap:
        raise ValueError("Synthetic incident document exactly matches a WikiText row")
    return {"rows": rows_compared, "unique_hashes": len(row_hashes)}


def prepare(
    model_dir,
    corpus_file,
    generator_source,
    license_file,
    reproduced_dir,
    wikitext_train,
    wikitext_test,
    preexposure_report,
    output,
):
    """Freeze every document before any model-weight load or loss evaluation."""
    source = provenance()
    assets = _validate_frozen_assets(model_dir, corpus_file, generator_source, license_file)
    reproduction = _validate_reproduction(corpus_file, reproduced_dir)
    preexposure = json.loads(Path(preexposure_report).read_text())
    if preexposure != {
        "corpus_sha256": EXPECTED_SPLIT_SHA256["test"],
        "documents": 50,
        "unique_incident_ids": 50,
        "unique_canonical_documents": 50,
        "exact_hash_overlap_wikitext_train_test": 0,
        "wikitext_rows_compared": 41076,
    }:
        raise ValueError("Pre-source-freeze exposure report changed")
    records, documents, hashes = _load_corpus(corpus_file)
    wikitext_comparison = _wikitext_overlap(hashes, wikitext_train, wikitext_test)
    if wikitext_comparison["rows"] != preexposure["wikitext_rows_compared"]:
        raise ValueError("WikiText comparison population changed")
    # model_dir is a hash-validated local snapshot and remote access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, trust_remote_code=False
    )
    rows = []
    for index, (record, document, digest) in enumerate(zip(records, documents, hashes, strict=True)):
        tokens = len(tokenizer(document, add_special_tokens=True)["input_ids"])
        if tokens < 2:
            raise ValueError("Every frozen document must contain a predicted target")
        rows.append(
            {
                "row": index,
                "incident_id": record["incident_id"],
                "sha256": digest,
                "tokens": tokens,
            }
        )
    manifest = {
        "protocol": PROTOCOL,
        **source,
        "model_asset_sha256": assets,
        "corpus": {
            "split_sha256": reproduction,
            "generator_commit": EXPECTED_GENERATOR_COMMIT,
            "generator_sha256": EXPECTED_GENERATOR_SHA256,
            "license": "MIT",
            "license_sha256": EXPECTED_LICENSE_SHA256,
            "generator_command": "generate_dataset(200, 50, 50, seed=0)",
            "reproduced_all_splits_exactly": True,
            "preexposure_compress_head": EXPECTED_PREEXPOSURE_HEAD,
            "preexposure_report_sha256": file_hash(preexposure_report),
            "prior_compress_hash_matches": 0,
            "exact_wikitext_hash_overlap": 0,
            "wikitext_rows_compared": wikitext_comparison["rows"],
            "wikitext_unique_row_hashes_compared": wikitext_comparison["unique_hashes"],
            "input_fields": list(INPUT_FIELDS),
            "documents": rows,
            "scope": "Small template-generated synthetic incident test split; not natural-corpus evidence",
        },
    }
    with Path(output).open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def validate_manifest(
    manifest,
    model_dir,
    corpus_file,
    generator_source,
    license_file,
    reproduced_dir,
    wikitext_train,
    wikitext_test,
    preexposure_report,
):
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Manifest protocol differs from frozen source")
    if manifest["model_asset_sha256"] != _validate_frozen_assets(
        model_dir, corpus_file, generator_source, license_file
    ):
        raise ValueError("Manifest model assets changed")
    if manifest["corpus"]["split_sha256"] != _validate_reproduction(
        corpus_file, reproduced_dir
    ):
        raise ValueError("Manifest corpus reproduction changed")
    if file_hash(preexposure_report) != manifest["corpus"]["preexposure_report_sha256"]:
        raise ValueError("Pre-source-freeze exposure report changed")
    records, documents, hashes = _load_corpus(corpus_file)
    comparison = _wikitext_overlap(hashes, wikitext_train, wikitext_test)
    expected_comparison = {
        "rows": manifest["corpus"]["wikitext_rows_compared"],
        "unique_hashes": manifest["corpus"]["wikitext_unique_row_hashes_compared"],
    }
    if comparison != expected_comparison:
        raise ValueError("WikiText comparison population changed")
    # model_dir is a hash-validated local snapshot and remote access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, trust_remote_code=False
    )
    rows = manifest["corpus"].get("documents")
    if not isinstance(rows, list) or len(rows) != len(documents):
        raise ValueError("Manifest must retain all 50 frozen documents")
    for index, (record, document, digest, frozen) in enumerate(
        zip(records, documents, hashes, rows, strict=True)
    ):
        expected = {
            "row": index,
            "incident_id": record["incident_id"],
            "sha256": digest,
            "tokens": len(tokenizer(document, add_special_tokens=True)["input_ids"]),
        }
        if frozen != expected:
            raise ValueError("Frozen synthetic document changed")
    return documents


def quality_gate(results):
    if set(results) != set(PROTOCOL["method_order"]):
        raise ValueError("Require every frozen method")
    counts = {row["predicted_tokens"] for row in results.values()}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("Methods must score identical positive target counts")
    for row in results.values():
        if (
            not math.isfinite(row["mean_nll"])
            or row["mean_nll"] < 0
            or not all(
                row[name]
                for name in (
                    "checkpoint_tensor_roundtrip",
                    "cpu_logits_exact_roundtrip",
                    "shifted_loss_verified",
                )
            )
        ):
            raise ValueError("Invalid method result or failed correctness gate")
    candidate = results["linear_embedding_int8"]
    ratio = math.exp(candidate["mean_nll"] - results["fp32"]["mean_nll"])
    bytes_exact = (
        candidate["persistent_tensor_bytes"] == PROTOCOL["expected_candidate_tensor_bytes"]
    )
    return {
        "candidate_fp32_perplexity_ratio": ratio,
        "candidate_tensor_bytes_exact": bytes_exact,
        "practical_gate": bytes_exact and ratio <= PROTOCOL["maximum_fp32_ppl_ratio"],
        "scope": "Second-domain synthetic engineering check; no natural-corpus generalization",
    }


def run(
    model_dir,
    corpus_file,
    generator_source,
    license_file,
    reproduced_dir,
    wikitext_train,
    wikitext_test,
    preexposure_report,
    manifest_file,
    output,
):
    source = provenance()
    manifest = json.loads(Path(manifest_file).read_text())
    if any(manifest[key] != source[key] for key in source):
        raise ValueError("Source changed after manifest freeze")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("This is an explicit CPU-only run; accelerator fallback must be disabled")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": PROTOCOL,
        **source,
        "manifest": manifest,
        "manifest_sha256": file_hash(manifest_file),
        "device_contract": {
            "requested": "cpu",
            "actual": "cpu",
            "quantization": "cpu",
            "accelerator_fallback": "not_applicable",
            "threads": 1,
            "checkpoint": "safetensors; validated schema before model mutation",
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": {
            name: version(name)
            for name in ("torch", "transformers", "safetensors", "pyarrow", "numpy")
        },
        "results": {},
        "complete": False,
        "error": None,
    }

    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write()
    try:
        documents = validate_manifest(
            manifest,
            model_dir,
            corpus_file,
            generator_source,
            license_file,
            reproduced_dir,
            wikitext_train,
            wikitext_test,
            preexposure_report,
        )
        torch.set_num_threads(1)
        torch.manual_seed(42)
        # model_dir was hash-validated by validate_manifest; remote access is disabled.
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_dir, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_dir,
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        check_ids = tokenizer(documents[0], return_tensors="pt")["input_ids"][:, :128]
        expected_count = sum(row["tokens"] - 1 for row in manifest["corpus"]["documents"])
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        ]
        assignments = dict.fromkeys(linear_names, "int8")

        def evaluate(name, temporary):
            candidate = copy.deepcopy(model)
            packed = name != "fp32"
            if packed:
                candidate = pack_model(
                    candidate,
                    QuantizationConfig(assignments, protect_output_head=False),
                    pack_embeddings=name == "linear_embedding_int8",
                )
            checkpoint = Path(temporary) / f"{name}.safetensors"
            hashes = state_hashes(candidate)
            if packed:
                save_packed(candidate, checkpoint)
            else:
                save_file(
                    {
                        key: value.detach().cpu().contiguous().clone()
                        for key, value in candidate.state_dict().items()
                    },
                    str(checkpoint),
                )
            with torch.inference_mode():
                before = candidate(input_ids=check_ids).logits
            stored_bytes = tensor_bytes(candidate)
            del candidate
            gc.collect()
            candidate = AutoModelForCausalLM.from_config(
                copy.deepcopy(model.config), attn_implementation="eager"
            ).eval()
            candidate.config.use_cache = False
            if packed:
                candidate = load_packed(candidate, checkpoint)
            else:
                candidate.load_state_dict(load_file(str(checkpoint)), strict=True)
            if hashes != state_hashes(candidate):
                raise ValueError("Save/reload changed tensor values")
            with torch.inference_mode():
                after = candidate(input_ids=check_ids).logits
                if not torch.equal(before, after):
                    raise ValueError("Save/reload changed CPU logits")
                manual = torch.nn.functional.cross_entropy(
                    after[:, :-1].float().reshape(-1, after.shape[-1]),
                    check_ids[:, 1:].reshape(-1),
                )
                labeled = candidate(input_ids=check_ids, labels=check_ids).loss
                if not torch.allclose(manual, labeled, atol=1e-5, rtol=1e-5):
                    raise ValueError("Shifted loss differs from model loss")
            per_document = []
            for row, document in zip(manifest["corpus"]["documents"], documents, strict=True):
                nll, count, _ = _negative_log_likelihood(
                    candidate,
                    tokenizer,
                    [document],
                    PROTOCOL["evaluation_context"],
                    PROTOCOL["evaluation_stride"],
                    "cpu",
                )
                if count != row["tokens"] - 1:
                    raise ValueError("Document target count mismatch")
                per_document.append(
                    {
                        "row": row["row"],
                        "incident_id": row["incident_id"],
                        "nll": nll,
                        "tokens": count,
                    }
                )
            total = sum(row["nll"] for row in per_document)
            count = sum(row["tokens"] for row in per_document)
            if count != expected_count:
                raise ValueError("Incomplete second-domain panel")
            metadata = {}
            if packed:
                with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
                    metadata = handle.metadata() or {}
            result = {
                "method": name,
                "persistent_tensor_bytes": stored_bytes,
                "checkpoint_sha256": file_hash(checkpoint),
                "checkpoint_format_version": metadata.get("format_version", "fp32"),
                "tensor_hash_aggregate": hashlib.sha256(
                    json.dumps(hashes, sort_keys=True).encode()
                ).hexdigest(),
                "checkpoint_tensor_roundtrip": True,
                "cpu_logits_exact_roundtrip": True,
                "shifted_loss_verified": True,
                "predicted_tokens": count,
                "negative_log_likelihood": total,
                "mean_nll": total / count,
                "perplexity": math.exp(total / count),
                "per_document": per_document,
            }
            checkpoint.unlink()
            del candidate, before, after, manual, labeled
            gc.collect()
            return result

        with tempfile.TemporaryDirectory() as temporary:
            for name in PROTOCOL["method_order"]:
                report["results"][name] = evaluate(name, temporary)
                write()
                print(f"{name}: {report['results'][name]['perplexity']}", flush=True)
        report["acceptance"] = quality_gate(report["results"])
        report["complete"] = True
        write()
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--generator-source", required=True)
    parser.add_argument("--license-file", required=True)
    parser.add_argument("--reproduced-dir", required=True)
    parser.add_argument("--wikitext-train", required=True)
    parser.add_argument("--wikitext-test", required=True)
    parser.add_argument("--preexposure-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    common = (
        args.model_dir,
        args.corpus_file,
        args.generator_source,
        args.license_file,
        args.reproduced_dir,
        args.wikitext_train,
        args.wikitext_test,
        args.preexposure_report,
    )
    if args.prepare:
        prepare(*common, args.manifest)
    elif args.output:
        run(*common, args.manifest, args.output)
    else:
        parser.error("--output is required for execution")
