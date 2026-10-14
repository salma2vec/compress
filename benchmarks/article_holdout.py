"""Deterministic article-group holdout, with no access to model predictions."""

import hashlib
import random


def article_ids(rows):
    """Assign by top-level WikiText heading before paragraph/token filtering."""
    current = None
    result = []
    for index, text in enumerate(rows):
        heading = text.strip()
        inner = heading[2:-2].strip()
        if (heading.startswith("= ") and heading.endswith(" =") and inner
                and not inner.startswith("=") and not inner.endswith("=")):
            current = index
        result.append(current)
    return result


def select_article_holdout(rows, tokenizer, exposed_rows, calibration_texts, *,
                           seed=1042, article_count=16, paragraphs_per_article=2):
    if article_count < 1 or paragraphs_per_article < 1:
        raise ValueError("Positive article and paragraph counts required")
    groups = article_ids(rows)
    if any(type(index) is not int or not 0 <= index < len(rows) for index in exposed_rows):
        raise ValueError("Exposed paragraph index out of range")
    if any(groups[index] is None for index in exposed_rows):
        raise ValueError("Exposed paragraph lacks an article group")
    excluded = {groups[index] for index in exposed_rows}
    known_texts = set(calibration_texts)
    candidates = {}
    seen = set()
    for index, text in enumerate(rows):
        article = groups[index]
        if (article is None or article in excluded or not text.strip()
                or text.lstrip().startswith("=") or text in known_texts or text in seen):
            continue
        seen.add(text)
        length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if 128 <= length <= 512:
            candidates.setdefault(article, []).append({
                "row": index, "article": article, "tokens": length,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
    eligible = sorted(article for article, paragraphs in candidates.items()
                      if len(paragraphs) >= paragraphs_per_article)
    if len(eligible) < article_count:
        raise ValueError("Insufficient unexposed article groups; do not relax after seeing losses")
    rng = random.Random(seed)
    chosen = rng.sample(eligible, article_count)
    selected = [row for article in chosen
                for row in rng.sample(candidates[article], paragraphs_per_article)]
    return {
        "seed": seed, "excluded_articles": sorted(excluded),
        "eligible_article_count": len(eligible), "selected_articles": chosen,
        "paragraphs": selected,
        "scope": "Article-balanced sampled test paragraphs, not standard full-test WikiText",
    }
