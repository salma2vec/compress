"""Compare CPU and accelerator logits for the first held-out 128-token window."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.pretrained import DATASET, DATASET_REVISION, MODEL, MODEL_REVISION, digest
from compress.compression.packed import pack_model
from compress.compression.quantization import QuantizationConfig


def run(report_path, output, device="mps"):
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable CPU fallback for accelerator validation")
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists")
    report = json.loads(Path(report_path).read_text())
    if (report["model_revision"], report["dataset_revision"]) != (MODEL_REVISION, DATASET_REVISION):
        raise ValueError("This diagnostic requires the pinned Pythia/WikiText revisions")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION,
                                              local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True, use_safetensors=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager",
    ).eval()
    model.config.use_cache = False
    path = hf_hub_download(DATASET, "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=DATASET_REVISION, token=False,
                           local_files_only=True)
    paragraph = report["splits"]["test"][0]
    text = parquet.read_table(path)["text"][paragraph["row"]].as_py()
    if digest(text.encode()) != paragraph["sha256"]:
        raise ValueError("Dataset row hash differs from recorded experiment")
    ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :128]
    result = {"model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
              "row": paragraph["row"], "tokens": ids.shape[1], "device": device,
              "mps_cpu_fallback_enabled": False, "atol": 1e-4, "rtol": 1e-4, "results": []}
    for method in ("fp32", "uniform_int8", "uniform_nf4"):
        candidate = copy.deepcopy(model)
        if method != "fp32":
            plan = {name: "int4" if method == "uniform_nf4" else "int8"
                    for name, module in candidate.named_modules() if isinstance(module, torch.nn.Linear)}
            candidate = pack_model(candidate, QuantizationConfig(plan, protect_output_head=False))
        with torch.inference_mode():
            cpu = candidate(input_ids=ids, labels=ids)
            cpu_logits, cpu_loss = cpu.logits.detach().clone(), float(cpu.loss)
            del cpu
            candidate.to(device)
            observed = candidate(input_ids=ids.to(device), labels=ids.to(device))
            logits, loss = observed.logits.cpu(), float(observed.loss)
            result["results"].append({
                "method": method, "cpu_loss": cpu_loss, "mps_loss" if device == "mps" else "device_loss": loss,
                "max_abs_logit_difference": float((cpu_logits - logits).abs().max()),
                "logits_allclose": bool(torch.allclose(cpu_logits, logits, atol=1e-4, rtol=1e-4)),
            })
        del candidate, observed, logits, cpu_logits
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["mps", "cuda"], default="mps")
    args = parser.parse_args()
    result = run(args.report, args.output, args.device)
    raise SystemExit(0 if all(row["logits_allclose"] for row in result["results"]) else 1)
