import json
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

app = typer.Typer(
    help="compress: activation-aware compression for transformer models.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def profile(
    model: str = typer.Option("gpt2", help="HuggingFace model name or local path."),
    data: str = typer.Option("", help="Path to a text file (one sample per line). Uses synthetic data if empty."),
    out: str = typer.Option("profile.json", help="Output path for the activation profile."),
    n_batches: int = typer.Option(50, help="Number of calibration inputs to process."),
):
    """Profile a model's activation patterns on calibration data.

    Attaches forward hooks to all attention and MLP layers, runs calibration
    inputs through the model, and saves a JSON activation profile.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from compress.profiling.activation_recorder import ActivationRecorder
    from compress.profiling.calibration_data import build_calibration_set

    console.print(f"[bold]Loading model:[/bold] {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)

    if data:
        with open(data) as f:
            texts = [line.strip() for line in f if line.strip()]
        console.print(f"Loaded {len(texts)} samples from {data}")
    else:
        texts = build_calibration_set(n=n_batches * 2, seed=42)
        console.print(f"Using synthetic calibration data ({len(texts)} samples)")

    console.print(f"Profiling with up to {n_batches} batches...")
    recorder = ActivationRecorder(lm)
    activation_profile = recorder.record(texts, tokenizer, n_batches=n_batches)

    profile_data = activation_profile.to_dict()
    with open(out, "w") as f:
        json.dump(profile_data, f, indent=2)

    console.print(f"[green]Profile saved to {out}[/green] ({len(profile_data)} layers profiled)")


@app.command()
def quantize(
    model: str = typer.Option("gpt2", help="HuggingFace model name or local path."),
    profile_path: str = typer.Option("profile.json", help="Path to activation profile JSON."),
    out: str = typer.Option("model_compressed/", help="Output directory for the compressed model."),
    default_dtype: str = typer.Option("int8", help="Default dtype for layers not in profile."),
):
    """Apply activation-aware mixed-precision quantization.

    Loads an activation profile, computes per-layer importance scores, assigns
    heuristic dtypes (fp16/int8/int4), and saves the compressed model weights.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from compress.profiling.activation_recorder import ActivationProfile
    from compress.profiling.importance import ImportanceScorer
    from compress.compression.quantization import ActivationAwareQuantizer, QuantizationConfig
    from compress.compression.mixed_precision import MixedPrecisionConfig

    console.print(f"[bold]Loading model:[/bold] {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)

    console.print(f"Loading profile from {profile_path}")
    with open(profile_path) as f:
        profile_data = json.load(f)
    activation_profile = ActivationProfile.from_dict(profile_data)

    scorer = ImportanceScorer(activation_profile)
    importance = scorer.layer_importance()
    mp_config = MixedPrecisionConfig.from_importance(importance)

    summary = mp_config.summary()
    table = Table(title="Quantization Plan")
    table.add_column("Dtype")
    table.add_column("Layers", justify="right")
    for dtype, count in summary.items():
        table.add_row(dtype, str(count))
    console.print(table)

    q_config = QuantizationConfig(layer_dtypes=mp_config.layer_dtypes, default_dtype=default_dtype)
    quantizer = ActivationAwareQuantizer(q_config)
    lm = quantizer.quantize_model(lm)

    import os
    os.makedirs(out, exist_ok=True)
    lm.save_pretrained(out)
    tokenizer.save_pretrained(out)
    console.print(f"[green]Fake-quantized floating-point model saved to {out}[/green]")


@app.command()
def evaluate(
    model: str = typer.Option("gpt2", help="Model name or path to compressed model directory."),
    data: str = typer.Option("", help="Path to evaluation text file (one sample per line)."),
    n_samples: int = typer.Option(50, help="Number of evaluation samples."),
    n_tokens: int = typer.Option(20, help="Tokens to generate per latency run."),
):
    """Evaluate a compressed model's perplexity and inference latency."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from compress.evaluation.perplexity import compute_perplexity
    from compress.evaluation.latency import measure_latency
    from compress.profiling.calibration_data import build_calibration_set

    console.print(f"[bold]Loading model:[/bold] {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)

    if data:
        with open(data) as f:
            texts = [line.strip() for line in f if line.strip()][:n_samples]
    else:
        texts = build_calibration_set(n=n_samples, seed=99)

    console.print(f"Computing perplexity on {len(texts)} samples...")
    ppl = compute_perplexity(lm, tokenizer, texts)
    console.print(f"Computing latency ({n_tokens} tokens, 3 runs)...")
    lat = measure_latency(lm, tokenizer, n_tokens=n_tokens, n_runs=3)

    table = Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Perplexity", f"{ppl:.2f}")
    table.add_row("Tokens/sec", f"{lat['tokens_per_sec']:.1f}")
    table.add_row("Mean latency (s)", f"{lat['mean_latency_s']:.3f}")
    console.print(table)


@app.command()
def benchmark(
    model: str = typer.Option("gpt2", help="HuggingFace model name to benchmark."),
    data: str = typer.Option("", help="Optional path to calibration data file."),
):
    """Run the full compression benchmark suite and print a comparison table."""
    from benchmarks.gpt2_compression import run_benchmark
    run_benchmark(model_name=model, data_path=data)


if __name__ == "__main__":
    app()
