"""Frozen fresh-panel validation for portable rowwise-INT8 embeddings."""

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

from benchmarks.article_holdout import article_ids, select_article_holdout
from benchmarks.failure_diagnosis import file_hash, state_hashes, validate_assets
from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import QuantizationConfig
from compress.evaluation.perplexity import _negative_log_likelihood

PROTOCOL = {
    "version": 1,
    "article_seed": 1043,
    "article_count": 8,
    "paragraphs_per_article": 2,
    "evaluation_context": 256,
    "evaluation_stride": 128,
    "maximum_fp32_ppl_ratio": 1.05,
    "expected_candidate_tensor_bytes": 71_062_592,
    "method_order": ["fp32", "linear_int8", "linear_embedding_int8"],
    "device": "cpu",
    "threads": 1,
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
        "protocol_sha256": file_hash(root / "docs/embedding-int8-validation-protocol.md"),
    }


def _verified_text(rows, record, label):
    index = record["row"]
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError(f"{label} row index out of range")
    text = rows[index]
    if hashlib.sha256(text.encode()).hexdigest() != record["sha256"]:
        raise ValueError(f"Changed {label} paragraph")
    return text


def prepare(model_dir, train_file, test_file, pilot_path, allocation_manifest_path, output):
    """Freeze a new article panel before any model-weight load or loss evaluation."""
    source = provenance()
    pilot_path, allocation_manifest_path = Path(pilot_path), Path(allocation_manifest_path)
    pilot = json.loads(pilot_path.read_text())
    allocation = json.loads(allocation_manifest_path.read_text())
    assets = validate_assets(model_dir, test_file, pilot)
    train_key = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
    if file_hash(train_file) != pilot["asset_sha256"][train_key]:
        raise ValueError("Cached training asset changed")
    assets[train_key] = file_hash(train_file)
    # model_dir is a hash-validated local snapshot and remote access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, trust_remote_code=False
    )
    train_rows = parquet.read_table(train_file)["text"].to_pylist()
    test_rows = parquet.read_table(test_file)["text"].to_pylist()

    exposed = []
    for record in pilot["splits"]["test"]:
        _verified_text(test_rows, record, "pilot evaluation")
        exposed.append(record["row"])
    for record in allocation["holdout"]["paragraphs"]:
        _verified_text(test_rows, record, "allocation evaluation")
        exposed.append(record["row"])
    known_calibration = [
        _verified_text(train_rows, record, "pilot calibration")
        for record in pilot["splits"]["train"]
    ]
    for records in allocation["calibration"].values():
        known_calibration.extend(
            _verified_text(train_rows, record, "allocation calibration") for record in records
        )
    holdout = select_article_holdout(
        test_rows,
        tokenizer,
        exposed,
        known_calibration,
        seed=PROTOCOL["article_seed"],
        article_count=PROTOCOL["article_count"],
        paragraphs_per_article=PROTOCOL["paragraphs_per_article"],
    )
    manifest = {
        "protocol": PROTOCOL,
        **source,
        "asset_sha256": assets,
        "pilot_report_sha256": file_hash(pilot_path),
        "allocation_manifest_sha256": file_hash(allocation_manifest_path),
        "holdout": holdout,
    }
    with Path(output).open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def validate_manifest(
    manifest, model_dir, train_file, test_file, pilot_path, allocation_manifest_path
):
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Manifest protocol differs from frozen source")
    if file_hash(pilot_path) != manifest["pilot_report_sha256"]:
        raise ValueError("Pilot report changed")
    if file_hash(allocation_manifest_path) != manifest["allocation_manifest_sha256"]:
        raise ValueError("Allocation manifest changed")
    for name, expected in manifest["asset_sha256"].items():
        path = (
            train_file
            if name.endswith("train-00000-of-00001.parquet")
            else test_file
            if name.endswith("test-00000-of-00001.parquet")
            else Path(model_dir) / name
        )
        if file_hash(path) != expected:
            raise ValueError(f"Changed cached asset: {name}")
    test_rows = parquet.read_table(test_file)["text"].to_pylist()
    # model_dir is a hash-validated local snapshot and remote access is disabled.
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, trust_remote_code=False
    )
    holdout = manifest.get("holdout", {})
    selected = holdout.get("selected_articles")
    paragraphs = holdout.get("paragraphs")
    if (
        holdout.get("seed") != PROTOCOL["article_seed"]
        or not isinstance(selected, list)
        or len(selected) != PROTOCOL["article_count"]
        or len(set(selected)) != len(selected)
        or not isinstance(paragraphs, list)
        or len(paragraphs) != PROTOCOL["article_count"] * PROTOCOL["paragraphs_per_article"]
    ):
        raise ValueError("Frozen holdout must contain exactly eight distinct articles and 16 rows")
    groups = article_ids(test_rows)
    rows_seen, hashes_seen, article_counts = set(), set(), {}
    for record in paragraphs:
        text = _verified_text(test_rows, record, "embedding validation")
        if (
            record["row"] in rows_seen
            or record["sha256"] in hashes_seen
            or groups[record["row"]] != record["article"]
            or len(tokenizer(text, add_special_tokens=True)["input_ids"]) != record["tokens"]
            or not 128 <= record["tokens"] <= 512
        ):
            raise ValueError("Frozen holdout contains a duplicate or changed paragraph")
        rows_seen.add(record["row"])
        hashes_seen.add(record["sha256"])
        article_counts[record["article"]] = article_counts.get(record["article"], 0) + 1
    if (
        set(selected) != set(article_counts)
        or set(article_counts.values()) != {PROTOCOL["paragraphs_per_article"]}
        or set(selected) & set(holdout.get("excluded_articles", []))
    ):
        raise ValueError("Frozen holdout article grouping changed")
    return test_rows


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
        "scope": "One model and fresh sampled WikiText-2 articles; storage/quality only",
    }


def run(
    model_dir, train_file, test_file, pilot_path, allocation_manifest_path, manifest_file, output
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
        test_rows = validate_manifest(
            manifest, model_dir, train_file, test_file, pilot_path, allocation_manifest_path
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
        check_text = test_rows[manifest["holdout"]["paragraphs"][0]["row"]]
        check_ids = tokenizer(check_text, return_tensors="pt")["input_ids"][:, :128]
        expected_count = sum(row["tokens"] - 1 for row in manifest["holdout"]["paragraphs"])
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
            documents = []
            for row in manifest["holdout"]["paragraphs"]:
                nll, count, _ = _negative_log_likelihood(
                    candidate,
                    tokenizer,
                    [test_rows[row["row"]]],
                    PROTOCOL["evaluation_context"],
                    PROTOCOL["evaluation_stride"],
                    "cpu",
                )
                if count != row["tokens"] - 1:
                    raise ValueError("Paragraph target count mismatch")
                documents.append(
                    {"row": row["row"], "article": row["article"], "nll": nll, "tokens": count}
                )
            total = sum(row["nll"] for row in documents)
            count = sum(row["tokens"] for row in documents)
            if count != expected_count:
                raise ValueError("Incomplete holdout")
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
                "per_document": documents,
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
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--pilot-report", required=True)
    parser.add_argument("--allocation-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.prepare:
        prepare(
            args.model_dir,
            args.train_file,
            args.test_file,
            args.pilot_report,
            args.allocation_manifest,
            args.manifest,
        )
    elif args.output:
        run(
            args.model_dir,
            args.train_file,
            args.test_file,
            args.pilot_report,
            args.allocation_manifest,
            args.manifest,
            args.output,
        )
    else:
        parser.error("--output is required for execution")
