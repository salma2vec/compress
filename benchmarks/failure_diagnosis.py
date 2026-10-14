"""Offline controlled interventions for the recorded Pythia rowwise-NF4 failure.

This reuses observed pilot paragraphs for diagnosis, not confirmatory evaluation.
It neither downloads assets nor changes the quantizers or tunes an allocation.
"""

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
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.pretrained import DATASET_REVISION, MODEL_REVISION
from compress.compression.packed import PackedLinear, tensor_bytes
from compress.evaluation.perplexity import _negative_log_likelihood

METHODS = ("fp32", "uniform_int8", "uniform_nf4", "nf4_body_int8_head", "fp32_body_nf4_head")
MODEL_FILES = (
    "config.json", "model.safetensors", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json",
)
DATASET_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
PROTOCOL = {
    "version": 1, "minimum_excess_nll": 1.0, "dominant_fraction": 0.80,
    "context": 256, "stride": 128, "parity_context": 128,
    "parity_atol": 1e-4, "parity_rtol": 1e-4,
    "head": "lm_head", "methods": list(METHODS),
    "purpose": "Exploratory diagnosis on the previously observed pilot paragraphs",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hashes(model):
    return {
        name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, tensor in model.state_dict().items()
    }


def validate_assets(model_dir, dataset_file, pilot):
    """Hash only the fixed expected local assets, before model construction."""
    if (pilot["model_revision"], pilot["dataset_revision"]) != (
        MODEL_REVISION, DATASET_REVISION
    ):
        raise ValueError("Unexpected model/dataset revision")
    paths = {name: Path(model_dir) / name for name in MODEL_FILES}
    paths[DATASET_FILE] = Path(dataset_file)
    for name, path in paths.items():
        if not path.is_file() or file_hash(path) != pilot["asset_sha256"][name]:
            raise ValueError(f"Missing or changed local asset: {name}")
    return {name: pilot["asset_sha256"][name] for name in paths}


def intervention_plans(model):
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if PROTOCOL["head"] not in names:
        raise ValueError("Expected an ordinary lm_head linear module")
    return {
        "fp32": {},
        "uniform_int8": dict.fromkeys(names, "int8"),
        "uniform_nf4": dict.fromkeys(names, "int4"),
        "nf4_body_int8_head": {
            name: "int8" if name == PROTOCOL["head"] else "int4" for name in names
        },
        "fp32_body_nf4_head": {PROTOCOL["head"]: "int4"},
    }


def apply_intervention(model, plan):
    """Replace only declared modules; omitted modules remain exactly FP32.

    This experimental partial-conversion path is separate from pack_model,
    whose default precision would otherwise also quantize the omitted body.
    """
    replacements = []
    for name, precision in plan.items():
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear) or precision not in {"int8", "int4"}:
            raise ValueError("Interventions require ordinary linear modules and INT8/NF4")
        if not bool(torch.isfinite(module.weight).all()):
            raise ValueError("Nonfinite original weight")
        replacements.append((name, PackedLinear(module, precision)))
    for name, packed in replacements:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, packed)
    return model


def residuals(original, candidate, plan):
    results = {}
    for name, precision in plan.items():
        weight = original.get_submodule(name).weight.detach().float()
        reconstructed = candidate.get_submodule(name).dequantize()
        difference = reconstructed - weight
        results[name] = {
            "precision": precision, "shape": list(weight.shape),
            "mean_squared_weight_error": float(difference.square().mean()),
            "max_absolute_weight_error": float(difference.abs().max()),
            "original_mean_square": float(weight.square().mean()),
        }
    return results


def restore_checkpoint(checkpoint, factory, plan, expected_hashes):
    """Restore a locally generated state into the same explicit intervention."""
    restored = apply_intervention(factory(), plan)
    restored.load_state_dict(load_file(str(checkpoint)), strict=True)
    if state_hashes(restored) != expected_hashes:
        raise ValueError("Checkpoint changed tensor contents")
    return restored.eval()


def diagnosis_gate(results):
    if len(results) != len(METHODS) or {r["method"] for r in results} != set(METHODS):
        raise ValueError("Diagnosis requires every intervention exactly once")
    if any(type(row["predicted_tokens"]) is not int for row in results):
        raise ValueError("Token counts must be integers")
    counts = {row["predicted_tokens"] for row in results}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("Interventions must score identical positive token counts")
    losses = {}
    for row in results:
        loss = row["negative_log_likelihood"] / row["predicted_tokens"]
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Invalid intervention likelihood")
        losses[row["method"]] = loss
    excess = losses["uniform_nf4"] - losses["fp32"]
    result = {"mean_nll": losses, "nf4_excess_nll": excess, "status": "not_reproduced"}
    if excess <= PROTOCOL["minimum_excess_nll"]:
        return result
    rescue = (losses["uniform_nf4"] - losses["nf4_body_int8_head"]) / excess
    isolated = (losses["fp32_body_nf4_head"] - losses["fp32"]) / excess
    passed = rescue >= PROTOCOL["dominant_fraction"] and isolated >= PROTOCOL["dominant_fraction"]
    return {
        **result, "status": "supported" if passed else "rejected",
        "head_rescue_fraction": rescue, "head_only_fraction": isolated,
        "interpretation": "Descriptive controlled interventions; interactions need not add to one",
    }


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout
    if status.strip():
        raise ValueError("Freeze and commit source before running the diagnosis")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "source_commit": revision, "source_clean": True,
        "harness_sha256": file_hash(__file__),
        "protocol_sha256": file_hash(root / "docs/failure-diagnosis-protocol.md"),
    }


def run(model_dir, dataset_file, pilot_path, output, device="cpu"):
    source = source_provenance()
    if device not in {"cpu", "mps"}:
        raise ValueError("This bounded diagnosis supports CPU or explicitly available MPS")
    if device == "mps" and (
        not torch.backends.mps.is_available()
        or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    ):
        raise ValueError("MPS must be available with CPU fallback disabled")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "protocol": PROTOCOL, **source,
        "pilot_report_sha256": file_hash(pilot_path), "device": device,
        "python": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: version(name) for name in
                             ("torch", "transformers", "safetensors", "pyarrow", "numpy")},
        "threads": 1, "mps_cpu_fallback_enabled": False, "dtype": "float32",
        "results": [], "complete": False,
    }

    def write_report():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        pilot = json.loads(Path(pilot_path).read_text())
        report["asset_sha256"] = validate_assets(model_dir, dataset_file, pilot)
        # model_dir is a hash-validated local snapshot and remote access is disabled.
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        rows = parquet.read_table(dataset_file)["text"]
        selected = pilot["splits"]["test"]
        if len(selected) != 16 or len({row["row"] for row in selected}) != 16:
            raise ValueError("Expected the 16 distinct recorded pilot paragraphs")
        texts = []
        for paragraph in selected:
            text = rows[paragraph["row"]].as_py()
            if hashlib.sha256(text.encode()).hexdigest() != paragraph["sha256"]:
                raise ValueError("Paragraph differs from recorded pilot")
            count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            if count != paragraph["tokens"]:
                raise ValueError("Paragraph token count differs from recorded pilot")
            texts.append(text)
        report["paragraphs"] = selected
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            str(model_dir), local_files_only=True, use_safetensors=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        plans = intervention_plans(model)
        if len(plans["uniform_nf4"]) != 25:
            raise ValueError("Expected the pinned model's 25 linear modules")
        report["model_revision"] = MODEL_REVISION
        report["dataset_revision"] = DATASET_REVISION
        report["plans"] = plans
        ids = tokenizer(texts[0], return_tensors="pt")["input_ids"][:, :PROTOCOL["parity_context"]]

        def factory():
            restored = AutoModelForCausalLM.from_config(
                copy.deepcopy(model.config), attn_implementation="eager"
            ).eval()
            restored.config.use_cache = False
            return restored

        with tempfile.TemporaryDirectory(dir=output) as directory:
            for method in METHODS:
                plan = plans[method]
                candidate = apply_intervention(copy.deepcopy(model), plan)
                weight_errors = residuals(model, candidate, plan)
                hashes = state_hashes(candidate)
                checkpoint = Path(directory) / "intervention.safetensors"
                save_file(candidate.state_dict(), str(checkpoint))
                with torch.inference_mode():
                    checked = candidate(input_ids=ids, labels=ids)
                    explicit_loss = torch.nn.functional.cross_entropy(
                        checked.logits[:, :-1].float().reshape(-1, checked.logits.shape[-1]),
                        ids[:, 1:].reshape(-1),
                    )
                    if not torch.allclose(checked.loss, explicit_loss, atol=1e-5, rtol=1e-5):
                        raise ValueError("Model loss differs from explicit shifted-token loss")
                    before = checked.logits.detach().clone()
                    if not bool(torch.isfinite(before).all()):
                        raise ValueError("Nonfinite CPU logits")
                    del checked, explicit_loss
                del candidate
                gc.collect()
                candidate = restore_checkpoint(checkpoint, factory, plan, hashes)
                stored_bytes = tensor_bytes(candidate)
                with torch.inference_mode():
                    after = candidate(input_ids=ids).logits
                    if not torch.equal(before, after):
                        raise ValueError("Reload changed CPU logits")
                    candidate.to(device)
                    measured = candidate(input_ids=ids.to(device)).logits.cpu()
                    if not bool(torch.isfinite(measured).all()):
                        raise ValueError("Nonfinite device logits")
                    parity = bool(torch.allclose(
                        after, measured, atol=PROTOCOL["parity_atol"],
                        rtol=PROTOCOL["parity_rtol"],
                    ))
                    parity_error = float((after - measured).abs().max())
                    if not parity:
                        raise ValueError(f"CPU/{device} logit parity failed for {method}")
                del before, after, measured
                documents = []
                for paragraph, text in zip(selected, texts, strict=True):
                    nll, count, _ = _negative_log_likelihood(
                        candidate, tokenizer, [text], PROTOCOL["context"],
                        PROTOCOL["stride"], device,
                    )
                    if count != paragraph["tokens"] - 1:
                        raise ValueError("Incomplete paragraph target coverage")
                    documents.append({"row": paragraph["row"], "negative_log_likelihood": nll,
                                      "predicted_tokens": count})
                nll = sum(row["negative_log_likelihood"] for row in documents)
                count = sum(row["predicted_tokens"] for row in documents)
                report["results"].append({
                    "method": method, "negative_log_likelihood": nll,
                    "predicted_tokens": count, "perplexity": math.exp(nll / count),
                    "persistent_tensor_bytes": stored_bytes, "per_document": documents,
                    "weight_residuals": weight_errors, "checkpoint_sha256": file_hash(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "state_tensor_sha256": hashes, "roundtrip_tensor_hashes_match": True,
                    "explicit_shifted_loss_verified": True,
                    "roundtrip_cpu_logits_exact": True, "cpu_device_parity_passed": parity,
                    "cpu_device_max_absolute_logit_difference": parity_error,
                })
                write_report()
                print(f"{method}: NLL/token={nll / count:.6f}, bytes={stored_bytes}", flush=True)
                checkpoint.unlink()
                del candidate
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
        report["diagnosis"] = diagnosis_gate(report["results"])
        report["complete"] = True
        write_report()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_report()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--pilot-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    run(args.model_dir, args.dataset_file, args.pilot_report, args.output, args.device)
