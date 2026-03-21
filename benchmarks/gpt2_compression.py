"""GPT-2 compression benchmark: 4 methods head-to-head.

Profiles activation patterns on a synthetic calibration corpus, then
evaluates 4 compression configurations on perplexity and inference latency:

  1. Baseline (FP32)
  2. Uniform INT8
  3. Activation-aware mixed precision (FP16/INT8/INT4 per layer)
  4. Mixed precision + head pruning (30% of attention layers dropped)

Run with:
    python benchmarks/gpt2_compression.py
or:
    compress benchmark
"""

import random
import copy
import torch
from rich.table import Table
from rich.console import Console


def run_benchmark(n_cal: int = 200, n_test: int = 20, n_batches: int = 20):
    """Execute the full benchmark suite and print results to stdout.

    Args:
        n_cal: Number of calibration samples.
        n_test: Number of test samples for perplexity evaluation.
        n_batches: Number of calibration batches to profile.
    """
    console = Console()
    console.print("[bold]compress benchmark: GPT-2 (124M)[/bold]\n")

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        console.print("[red]transformers not installed. Run: pip install transformers[/red]")
        return

    console.print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Synthetic calibration corpus — deterministic, no external data dependency
    rng = random.Random(42)
    templates = [
        "The study of",
        "Recent advances in",
        "Machine learning has",
        "In this paper we",
        "Analysis shows that",
        "Experimental results for",
        "The proposed approach to",
        "Neural network models and",
    ]
    cal_texts = [
        rng.choice(templates) + f" topic {i} reveals interesting patterns and significant results."
        for i in range(n_cal)
    ]
    test_texts = cal_texts[:n_test]

    methods = [
        ("Baseline (FP32)", None, None),
        ("Uniform INT8", "int8", None),
        ("Activation-aware mixed", "mixed", None),
        ("Mixed + head pruning (30%)", "mixed", 0.7),
    ]

    table = Table(title="Compression Benchmark Results")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Perplexity", justify="right")
    table.add_column("Tokens/sec", justify="right")
    table.add_column("PPL delta", justify="right")

    baseline_ppl = None
    results = []

    for name, quant_mode, keep_frac in methods:
        console.print(f"  Running: [bold]{name}[/bold]")

        console.print("    Loading model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        if quant_mode is not None:
            from compress.profiling.activation_recorder import ActivationRecorder
            from compress.profiling.importance import ImportanceScorer
            from compress.compression.quantization import ActivationAwareQuantizer, QuantizationConfig

            console.print("    Profiling activations...")
            recorder = ActivationRecorder(model)
            profile = recorder.record(cal_texts, tokenizer, n_batches=n_batches)
            scorer = ImportanceScorer(profile)
            importance = scorer.layer_importance()

            if quant_mode == "int8":
                layer_dtypes = {k: "int8" for k in importance}
            else:
                from compress.compression.mixed_precision import MixedPrecisionConfig
                mp = MixedPrecisionConfig.from_importance(importance)
                layer_dtypes = mp.layer_dtypes

            config = QuantizationConfig(layer_dtypes=layer_dtypes)
            quantizer = ActivationAwareQuantizer(config)
            console.print("    Quantizing...")
            model = quantizer.quantize_model(model)

            if keep_frac is not None:
                from compress.compression.pruning import HeadPruner
                pruner = HeadPruner(keep_fraction=keep_frac)
                console.print(f"    Pruning to {int(keep_frac * 100)}% of attention layers...")
                model = pruner.prune_heads(model, importance)
                report = pruner.pruning_report()
                console.print(f"    Pruned {report['n_pruned']} layers")

        from compress.evaluation.perplexity import compute_perplexity
        from compress.evaluation.latency import measure_latency

        console.print("    Evaluating perplexity...")
        ppl = compute_perplexity(model, tokenizer, test_texts)

        console.print("    Measuring latency...")
        lat = measure_latency(model, tokenizer, n_tokens=20, n_runs=3, warmup_runs=1)

        if baseline_ppl is None:
            baseline_ppl = ppl

        delta = f"+{(ppl - baseline_ppl) / baseline_ppl * 100:.1f}%" if baseline_ppl and ppl != baseline_ppl else "—"
        table.add_row(name, f"{ppl:.1f}", f"{lat['tokens_per_sec']:.1f}", delta)
        results.append({"method": name, "ppl": ppl, "latency": lat})
        console.print(f"    PPL={ppl:.1f}, {lat['tokens_per_sec']:.1f} tok/s\n")

    console.print(table)

    # Print scorer summary for the last profiled method
    console.print("\n[dim]Baseline perplexity:[/dim]", f"{baseline_ppl:.1f}" if baseline_ppl else "N/A")
    return results


if __name__ == "__main__":
    run_benchmark()
