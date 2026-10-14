"""Pinned Pythia/WikiText experiment with paired equal-storage controls.

Downloads public safetensors/Parquet assets only. No remote code or paid inference.
"""
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
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as parquet
import torch
import transformers
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from compress.compression.packed import load_packed, pack_model, save_packed, tensor_bytes
from compress.compression.quantization import QuantizationConfig
from compress.evaluation.perplexity import _negative_log_likelihood
from compress.profiling.activation_recorder import ActivationRecorder

MODEL = "EleutherAI/pythia-70m"
MODEL_REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def select_paragraphs(rows, tokenizer, count, seed):
    """Sample complete paragraphs of 128–512 tokens, not headings or fragments."""
    eligible = []
    seen = set()
    for index, text in enumerate(rows):
        if not text.strip() or text.lstrip().startswith("=") or text in seen:
            continue
        seen.add(text)
        length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if 128 <= length <= 512:
            eligible.append((index, text, length))
    if count < 1 or len(eligible) < count:
        raise ValueError("Not enough eligible paragraphs for requested sample")
    return random.Random(seed).sample(eligible, count)


def precision_plans(model, scores, shuffle_seeds=(11, 23, 37)):
    """Rank within shape groups; half INT8, half NF4, singleton groups INT8.

    Fixed allocation avoids tuning thresholds against evaluation text. Shuffling
    within identical weight/bias shapes preserves actual packed tensor bytes.
    """
    groups = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name not in scores or not math.isfinite(scores[name]):
                raise ValueError(f"Missing or nonfinite importance for {name}")
            key = (tuple(module.weight.shape), module.bias is not None)
            groups.setdefault(key, []).append(name)
    if not groups:
        raise ValueError("No linear modules")
    mixed = {}
    for group in groups.values():
        ranked = sorted(group, key=lambda name: (-scores[name], name))
        mixed.update({name: "int8" if i < (len(ranked) + 1) // 2 else "int4"
                      for i, name in enumerate(ranked)})
    methods = {"fp32": None, "uniform_int8": {n: "int8" for n in mixed},
               "uniform_nf4": {n: "int4" for n in mixed}, "ranked_mixed": mixed}
    for seed in shuffle_seeds:
        rng = random.Random(seed)
        shuffled = dict(mixed)
        for group in groups.values():
            values = [mixed[name] for name in group]
            rng.shuffle(values)
            shuffled.update(zip(group, values))
        methods[f"shuffled_{seed}"] = shuffled
    return methods


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run(output, device="cpu", seed=42, calibration_count=32, evaluation_count=16):
    if device == "mps" and (not torch.backends.mps.is_available()
                            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
        raise ValueError("MPS must be available with CPU fallback disabled")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    source = snapshot_download(MODEL, revision=MODEL_REVISION,
                               allow_patterns=["*.json", "*.safetensors"], token=False)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    model.config.use_cache = False
    selected = {}
    dataset_files = {}
    for split, count, split_seed in (("train", calibration_count, seed),
                                     ("test", evaluation_count, seed + 1)):
        name = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        path = hf_hub_download(DATASET, name, repo_type="dataset", revision=DATASET_REVISION,
                               token=False)
        dataset_files[name] = digest(Path(path).read_bytes())
        selected[split] = select_paragraphs(parquet.read_table(path)["text"].to_pylist(),
                                            tokenizer, count, split_seed)
    if {text for _, text, _ in selected["train"]} & {text for _, text, _ in selected["test"]}:
        raise ValueError("Calibration/evaluation overlap")
    report = {
        "schema_version": 1, "model": MODEL, "model_revision": MODEL_REVISION,
        "dataset": DATASET, "dataset_revision": DATASET_REVISION,
        "asset_sha256": {**dataset_files, **{p.name: digest(p.read_bytes())
                          for p in Path(source).iterdir() if p.is_file()}},
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "device": device, "threads": 1, "seed": seed, "dtype": "float32",
        "mps_cpu_fallback_enabled": False, "attention": "eager", "use_cache": False,
        "calibration_context": 128, "evaluation_context": 256, "evaluation_stride": 128,
        "policy": "Within each identical weight/bias shape, highest magnitude*density half INT8, remainder NF4; singleton INT8.",
        "scope": "Small sampled WikiText-2 paragraphs; not the standard concatenated full-test benchmark. MPS is not CUDA.",
        "splits": {split: [{"row": i, "sha256": digest(text.encode()), "tokens": n}
                            for i, text, n in rows] for split, rows in selected.items()},
        "results": [],
    }
    # Calibration on CPU avoids accelerator reduction differences changing plans.
    profile = ActivationRecorder(model).record([t for _, t, _ in selected["train"]],
                                                tokenizer, n_batches=calibration_count)
    profile_data = profile.to_dict()
    scores = {name: s.mean_magnitude * (1 - s.sparsity) for name, s in profile.layers.items()}
    report["activation_profile"] = profile_data
    plans = precision_plans(model, scores)
    report["method_order"] = list(plans)
    # Fixed shuffle makes timing-order effects inspectable without favoring a method.
    random.Random(seed).shuffle(report["method_order"])
    evaluation = [text for _, text, _ in selected["test"]]
    timing_ids = tokenizer(evaluation[0], return_tensors="pt")["input_ids"][:, :128].to(device)
    report["timing_input_tokens"] = timing_ids.shape[1]
    report["timing_warmup"] = 3
    report["timing_repeats"] = 7
    with tempfile.TemporaryDirectory() as tmp:
        for name in report["method_order"]:
            assignment = plans[name]
            candidate = copy.deepcopy(model)
            checkpoint = Path(tmp) / f"{name}.safetensors"
            if assignment is None:
                save_file(candidate.state_dict(), str(checkpoint))
            else:
                candidate = pack_model(candidate, QuantizationConfig(assignment))
                save_packed(candidate, checkpoint)
            state_hashes = {k: digest(v.detach().cpu().contiguous().numpy().tobytes())
                            for k, v in candidate.state_dict().items()}
            del candidate
            gc.collect()
            candidate = AutoModelForCausalLM.from_config(copy.deepcopy(model.config),
                                                        attn_implementation="eager")
            if assignment is None:
                candidate.load_state_dict(load_file(str(checkpoint)), strict=True)
            else:
                candidate = load_packed(candidate, checkpoint)
            restored_hashes = {k: digest(v.detach().cpu().contiguous().numpy().tobytes())
                              for k, v in candidate.state_dict().items()}
            if state_hashes != restored_hashes:
                raise ValueError(f"Checkpoint changed tensors: {name}")
            stored_bytes = tensor_bytes(candidate)
            candidate.to(device).eval()
            doc_scores = []
            for row, text, _ in selected["test"]:
                nll, count, _ = _negative_log_likelihood(candidate, tokenizer, [text], 256, 128, device)
                doc_scores.append({"row": row, "negative_log_likelihood": nll, "tokens": count})
            timings = []
            with torch.inference_mode():
                # Independently verify the model's label-shift loss on a short window.
                check = candidate(input_ids=timing_ids, labels=timing_ids)
                manual_loss = torch.nn.functional.cross_entropy(
                    check.logits[:, :-1].float().reshape(-1, check.logits.shape[-1]),
                    timing_ids[:, 1:].reshape(-1))
                if not torch.allclose(check.loss, manual_loss, atol=1e-5, rtol=1e-5):
                    raise ValueError("Model loss does not match independent shifted-token loss")
                del check, manual_loss
                for i in range(10):
                    synchronize(device)
                    start = time.perf_counter()
                    candidate(input_ids=timing_ids)
                    synchronize(device)
                    if i >= 3:
                        timings.append(time.perf_counter() - start)
            total_nll = sum(d["negative_log_likelihood"] for d in doc_scores)
            count = sum(d["tokens"] for d in doc_scores)
            report["results"].append({
                "method": name, "assignments": assignment, "checkpoint_tensor_roundtrip": True,
                "checkpoint_sha256": digest(checkpoint.read_bytes()),
                "checkpoint_bytes": checkpoint.stat().st_size, "persistent_tensor_bytes": stored_bytes,
                "negative_log_likelihood": total_nll, "predicted_tokens": count,
                "perplexity": math.exp(total_nll / count), "per_document": doc_scores,
                "prefill_median_seconds": statistics.median(timings), "prefill_seconds": timings,
            })
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{name}: perplexity={math.exp(total_nll / count):.4f}, tensor_bytes={stored_bytes}", flush=True)
            del candidate
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    rows = {r["method"]: r for r in report["results"]}
    if any(rows[n]["persistent_tensor_bytes"] != rows["ranked_mixed"]["persistent_tensor_bytes"]
           for n in rows if n.startswith("shuffled_")):
        raise ValueError("Matched-storage control violated")
    report["matched_storage_verified"] = True
    report["complete"] = True
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--evaluation-count", type=int, default=16)
    args = parser.parse_args()
    run(args.output, args.device, args.seed, args.calibration_count, args.evaluation_count)
