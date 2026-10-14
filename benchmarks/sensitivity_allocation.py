"""Frozen calibration-only allocation study on cached Pythia/WikiText assets."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.article_holdout import select_article_holdout
from benchmarks.failure_diagnosis import file_hash, state_hashes, validate_assets
from benchmarks.pretrained import precision_plans, select_paragraphs
from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import QuantizationConfig
from compress.compression.sensitivity import equal_shape_plan, module_sensitivity
from compress.evaluation.perplexity import _negative_log_likelihood
from compress.profiling.activation_recorder import ActivationRecorder

PROTOCOL = {
    "version": 1, "calibration_seeds": [42, 142, 242, 342, 442],
    "calibration_paragraphs": 8, "calibration_context": 128,
    "article_seed": 1042, "article_count": 16, "paragraphs_per_article": 2,
    "evaluation_context": 256, "evaluation_stride": 128,
    "shuffle_seeds": [11, 23, 37], "minimum_nll_gain": 0.01,
    "maximum_nll_regression": 0.02, "minimum_passing_seeds": 4,
    "maximum_fp32_ppl_ratio": 1.05, "maximum_tensor_bytes": 143409728,
    "head": "lm_head", "quantization_device": "cpu",
}


def provenance():
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True,
                           capture_output=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("Freeze and commit source before preparing or running this study")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    return {"source_commit": revision, "source_clean": True,
            "harness_sha256": file_hash(__file__),
            "protocol_sha256": file_hash(root / "docs/sensitivity-allocation-protocol.md")}


def prepare(model_dir, train_file, test_file, pilot_path, output):
    """Freeze article/paragraph identities without loading a language model."""
    source = provenance()
    pilot = json.loads(Path(pilot_path).read_text())
    assets = validate_assets(model_dir, test_file, pilot)
    train_key = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
    if file_hash(train_file) != pilot["asset_sha256"][train_key]:
        raise ValueError("Calibration asset changed")
    assets[train_key] = file_hash(train_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    train_rows = parquet.read_table(train_file)["text"].to_pylist()
    test_rows = parquet.read_table(test_file)["text"].to_pylist()
    calibration = {str(seed): select_paragraphs(train_rows, tokenizer,
                    PROTOCOL["calibration_paragraphs"], seed)
                   for seed in PROTOCOL["calibration_seeds"]}
    for row in pilot["splits"]["test"]:
        if hashlib.sha256(test_rows[row["row"]].encode()).hexdigest() != row["sha256"]:
            raise ValueError("Old observed paragraph changed")
    prior_calibration = []
    for row in pilot["splits"]["train"]:
        text = train_rows[row["row"]]
        if hashlib.sha256(text.encode()).hexdigest() != row["sha256"]:
            raise ValueError("Old calibration paragraph changed")
        prior_calibration.append(text)
    heldout = select_article_holdout(
        test_rows, tokenizer, [row["row"] for row in pilot["splits"]["test"]],
        prior_calibration + [text for rows in calibration.values() for _, text, _ in rows],
        seed=PROTOCOL["article_seed"], article_count=PROTOCOL["article_count"],
        paragraphs_per_article=PROTOCOL["paragraphs_per_article"],
    )
    manifest = {
        "protocol": PROTOCOL, **source, "asset_sha256": assets,
        "pilot_report_sha256": file_hash(pilot_path), "holdout": heldout,
        "calibration": {seed: [{"row": i, "tokens": count,
                               "sha256": hashlib.sha256(text.encode()).hexdigest()}
                              for i, text, count in rows] for seed, rows in calibration.items()},
    }
    # Refuse to overwrite a frozen manifest, including an existing symlink.
    with Path(output).open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def validate_manifest(manifest, model_dir, train_file, test_file):
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Manifest protocol differs from frozen source")
    for name, expected in manifest["asset_sha256"].items():
        path = (train_file if name.endswith("train-00000-of-00001.parquet") else
                test_file if name.endswith("test-00000-of-00001.parquet") else Path(model_dir) / name)
        if file_hash(path) != expected:
            raise ValueError(f"Changed cached asset: {name}")
    train_rows = parquet.read_table(train_file)["text"].to_pylist()
    test_rows = parquet.read_table(test_file)["text"].to_pylist()
    for records, rows in [(records, train_rows) for records in manifest["calibration"].values()] + [
        (manifest["holdout"]["paragraphs"], test_rows)
    ]:
        for record in records:
            if hashlib.sha256(rows[record["row"]].encode()).hexdigest() != record["sha256"]:
                raise ValueError("Changed frozen paragraph")
    return train_rows, test_rows


def quality_gate(seed_results, fp32_loss):
    if ({row["seed"] for row in seed_results} != set(PROTOCOL["calibration_seeds"])
            or len(seed_results) != len(PROTOCOL["calibration_seeds"])):
        raise ValueError("Require all five calibration seeds")
    passing, practical, no_regression = 0, 0, True
    details = []
    for row in seed_results:
        results = row["methods"]
        names = ["sensitivity", "magnitude", "shuffled_11", "shuffled_23", "shuffled_37"]
        if set(results) != set(names):
            raise ValueError("Missing matched-budget comparator")
        sizes = {results[name]["persistent_tensor_bytes"] for name in names}
        counts = {results[name]["predicted_tokens"] for name in names}
        if len(sizes) != 1 or len(counts) != 1 or next(iter(counts)) <= 0:
            raise ValueError("Policies must have identical bytes and scored targets")
        losses = {name: results[name]["mean_nll"] for name in names}
        if not math.isfinite(fp32_loss) or any(not math.isfinite(x) or x < 0 for x in losses.values()):
            raise ValueError("Invalid held-out loss")
        gain = losses["magnitude"] - losses["sensitivity"]
        shuffled = statistics.median(losses[f"shuffled_{seed}"] for seed in PROTOCOL["shuffle_seeds"])
        passed = gain >= PROTOCOL["minimum_nll_gain"] and losses["sensitivity"] < shuffled
        ratio = math.exp(losses["sensitivity"] - fp32_loss)
        useful = (ratio <= PROTOCOL["maximum_fp32_ppl_ratio"] and
                  next(iter(sizes)) <= PROTOCOL["maximum_tensor_bytes"])
        passing += passed
        practical += useful
        no_regression &= gain >= -PROTOCOL["maximum_nll_regression"]
        details.append({"seed": row["seed"], "nll_gain_over_magnitude": gain,
                        "nll_gain_over_median_shuffle": shuffled - losses["sensitivity"],
                        "fp32_perplexity_ratio": ratio, "score_pass": passed,
                        "practical_pass": useful})
    return {"per_seed": details, "score_passing_seeds": passing,
            "practical_passing_seeds": practical, "no_large_nll_regression": no_regression,
            "score_gate": passing >= PROTOCOL["minimum_passing_seeds"] and no_regression,
            "practical_gate": practical >= PROTOCOL["minimum_passing_seeds"],
            "scope": "Descriptive one-model pilot; seeds share held-out articles"}


def run(model_dir, train_file, test_file, manifest_file, output, device="cpu"):
    source = provenance()
    manifest = json.loads(Path(manifest_file).read_text())
    if any(manifest[key] != source[key] for key in source):
        raise ValueError("Source changed after manifest freeze")
    if device not in {"cpu", "mps"} or (device == "mps" and (
        not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    )):
        raise ValueError("Require CPU or available MPS with fallback disabled")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": PROTOCOL, **source, "manifest": manifest,
              "manifest_sha256": file_hash(manifest_file), "device": device,
              "platform": platform.platform(), "python": platform.python_version(),
              "package_versions": {name: version(name) for name in
                                   ("torch", "transformers", "safetensors", "pyarrow", "numpy", "huggingface-hub")},
              "threads": 1, "mps_cpu_fallback_enabled": False, "references": {},
              "seeds": [], "complete": False}

    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        train_rows, test_rows = validate_manifest(manifest, model_dir, train_file, test_file)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        head = PROTOCOL["head"]
        expected_count = sum(row["tokens"] - 1 for row in manifest["holdout"]["paragraphs"])
        # All five policies are frozen using calibration alone before any held-out
        # loss is computed. The report retains the full scores and assignments.
        for seed in PROTOCOL["calibration_seeds"]:
            texts = [train_rows[row["row"]] for row in manifest["calibration"][str(seed)]]
            profile = ActivationRecorder(model.cpu()).record(texts, tokenizer, n_batches=len(texts))
            magnitude = {name: stats.mean_magnitude * (1 - stats.sparsity)
                         for name, stats in profile.layers.items()}
            calibration_ids = [tokenizer(text, return_tensors="pt")["input_ids"][:, :128].to(device)
                               for text in texts]
            scores = module_sensitivity(model.to(device), calibration_ids, output_head_names=(head,))
            model.cpu()
            del calibration_ids
            if device == "mps":
                torch.mps.empty_cache()
            plans = precision_plans(model, magnitude, PROTOCOL["shuffle_seeds"])
            assignments = {"magnitude": plans["ranked_mixed"], **{
                f"shuffled_{shuffle}": plans[f"shuffled_{shuffle}"] for shuffle in PROTOCOL["shuffle_seeds"]}}
            assignments["sensitivity"] = equal_shape_plan(
                model, {name: row["benefit_per_byte"] for name, row in scores.items()},
                output_head_names=(head,),
            ).layer_dtypes
            if any(plan[head] != "int8" for plan in assignments.values()):
                raise ValueError("Every allocation must protect the output head")
            record = {"seed": seed, "activation_profile": profile.to_dict(),
                      "sensitivity_scores": scores, "assignments": assignments, "methods": {}}
            report["seeds"].append(record)
            write()
            print(f"calibration seed {seed} frozen", flush=True)

        def evaluate(name, assignment, temporary):
            candidate = copy.deepcopy(model)
            if assignment is not None:
                candidate = pack_model(candidate, QuantizationConfig(assignment))
            checkpoint = Path(temporary) / "candidate.safetensors"
            hashes = state_hashes(candidate)
            if assignment is None:
                save_file(candidate.state_dict(), str(checkpoint))
            else:
                save_packed(candidate, checkpoint)
            check_ids = tokenizer(train_rows[manifest["calibration"]["42"][0]["row"]],
                                  return_tensors="pt")["input_ids"][:, :128]
            with torch.inference_mode():
                before = candidate(input_ids=check_ids).logits
            stored_bytes = tensor_bytes(candidate)
            del candidate
            gc.collect()
            candidate = AutoModelForCausalLM.from_config(copy.deepcopy(model.config),
                                                        attn_implementation="eager").eval()
            if assignment is None:
                candidate.load_state_dict(load_file(str(checkpoint)), strict=True)
            else:
                candidate = load_packed(candidate, checkpoint)
            if hashes != state_hashes(candidate):
                raise ValueError("Save/reload changed tensor values")
            with torch.inference_mode():
                after = candidate(input_ids=check_ids).logits
                if not torch.equal(before, after):
                    raise ValueError("Save/reload changed CPU logits")
                candidate.to(device)
                parity = candidate(input_ids=check_ids.to(device))
                if not torch.allclose(after, parity.logits.cpu(), atol=1e-4, rtol=1e-4):
                    raise ValueError("CPU/accelerator logit parity failed")
                manual = torch.nn.functional.cross_entropy(
                    parity.logits[:, :-1].float().reshape(-1, parity.logits.shape[-1]),
                    check_ids[:, 1:].to(device).reshape(-1))
                labeled = candidate(input_ids=check_ids.to(device), labels=check_ids.to(device)).loss
                if not torch.allclose(manual, labeled, atol=1e-5, rtol=1e-5):
                    raise ValueError("Shifted loss differs from model loss")
            del before, after, parity, manual, labeled
            documents = []
            for row in manifest["holdout"]["paragraphs"]:
                nll, count, _ = _negative_log_likelihood(candidate, tokenizer,
                    [test_rows[row["row"]]], PROTOCOL["evaluation_context"],
                    PROTOCOL["evaluation_stride"], device)
                if count != row["tokens"] - 1:
                    raise ValueError("Paragraph target count mismatch")
                documents.append({"row": row["row"], "article": row["article"],
                                  "nll": nll, "tokens": count})
            total = sum(row["nll"] for row in documents)
            count = sum(row["tokens"] for row in documents)
            if count != expected_count:
                raise ValueError("Incomplete holdout")
            result = {"method": name, "persistent_tensor_bytes": stored_bytes,
                      "checkpoint_sha256": file_hash(checkpoint),
                      "tensor_hash_aggregate": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                      "checkpoint_tensor_roundtrip": True, "cpu_logits_exact_roundtrip": True,
                      "cpu_device_logit_parity": True, "shifted_loss_verified": True,
                      "predicted_tokens": count, "negative_log_likelihood": total,
                      "mean_nll": total / count, "perplexity": math.exp(total / count),
                      "per_document": documents}
            checkpoint.unlink()
            del candidate
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            return result

        with tempfile.TemporaryDirectory() as temporary:
            names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
            for name, assignment in [("fp32", None), ("uniform_int8", dict.fromkeys(names, "int8")),
                                     ("nf4_body_int8_head", {name: "int8" if name == head else "int4"
                                                             for name in names})]:
                report["references"][name] = evaluate(name, assignment, temporary)
                write()
            for record in report["seeds"]:
                order = list(record["assignments"])
                random.Random(record["seed"]).shuffle(order)
                record["evaluation_order"] = order
                for name in order:
                    record["methods"][name] = evaluate(name, record["assignments"][name], temporary)
                    write()
                    print(f"seed {record['seed']} {name}: {record['methods'][name]['perplexity']}", flush=True)
        report["acceptance"] = quality_gate(report["seeds"], report["references"]["fp32"]["mean_nll"])
        report["complete"] = True
        write()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pilot-report", default="benchmarks/results/pythia70m-wikitext2-mps-seed42.json")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.model_dir, args.train_file, args.test_file, args.pilot_report, args.manifest)
    elif args.output:
        run(args.model_dir, args.train_file, args.test_file, args.manifest, args.output, args.device)
    else:
        parser.error("--output is required for execution")
