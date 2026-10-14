import random


def build_calibration_set(
    n: int = 500,
    seed: int = 42,
    source: str = "wikitext",
    min_length: int = 50,
) -> list[str]:
    """Load or generate a calibration text corpus.

    Attempts to load from the Wikitext-103 dataset via HuggingFace datasets.
    Falls back to a synthetic corpus if the dataset is unavailable or the
    datasets library is not installed.

    Args:
        n: Number of samples to return.
        seed: Random seed for reproducibility.
        source: Dataset source identifier (currently only "wikitext" is used).
        min_length: Minimum character length for a sample to be included.

    Returns:
        List of text strings suitable for calibration forward passes.
    """
    rng = random.Random(seed)

    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        )
        texts = [row["text"] for row in ds if len(row["text"]) > min_length]
        if len(texts) >= n:
            return rng.sample(texts, n)
        return texts
    except Exception:
        pass

    return _synthetic_calibration_set(n=n, seed=seed)


def _synthetic_calibration_set(n: int = 500, seed: int = 42) -> list[str]:
    """Generate a synthetic calibration corpus covering diverse text patterns."""
    rng = random.Random(seed)
    templates = [
        "The study of {topic} has revealed several important findings regarding {aspect}.",
        "Recent advances in {topic} demonstrate that {aspect} plays a central role.",
        "Machine learning models trained on {topic} often exhibit {aspect}.",
        "In this work, we examine the relationship between {topic} and {aspect}.",
        "Experimental results confirm that {aspect} is critical for {topic}.",
        "The proposed method for {topic} achieves state-of-the-art performance on {aspect}.",
        "Analysis of {topic} reveals that {aspect} accounts for most of the variance.",
        "Neural networks applied to {topic} must carefully handle {aspect}.",
        "Understanding {topic} requires careful attention to {aspect} at multiple scales.",
        "Benchmarks comparing {topic} approaches show significant differences in {aspect}.",
    ]
    topics = [
        "natural language processing", "computer vision", "speech recognition",
        "graph neural networks", "reinforcement learning", "knowledge distillation",
        "model compression", "transfer learning", "few-shot learning", "meta-learning",
        "attention mechanisms", "transformer architectures", "tokenization strategies",
        "data augmentation", "regularization techniques",
    ]
    aspects = [
        "computational efficiency", "generalization performance", "training stability",
        "inference latency", "memory footprint", "calibration accuracy",
        "representation quality", "gradient flow", "loss landscape geometry",
        "emergent capabilities", "layer-wise behavior", "activation patterns",
        "weight distribution", "attention head specialization", "positional encoding",
    ]
    samples = []
    for i in range(n):
        template = rng.choice(templates)
        text = template.format(topic=rng.choice(topics), aspect=rng.choice(aspects))
        text += f" ({i})"
        samples.append(text)
    return samples
