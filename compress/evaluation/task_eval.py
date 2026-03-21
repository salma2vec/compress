import torch
from typing import Optional, Callable


class TaskEvaluator:
    """Evaluate compressed model quality on downstream task metrics.

    Provides utilities for classification accuracy, generation quality,
    and custom metric evaluation. Designed to work with HuggingFace models
    and standard NLP task formats.

    Args:
        model: The model to evaluate.
        tokenizer: Compatible tokenizer.
        device: Evaluation device. Defaults to model's current device.
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            try:
                device = next(model.parameters()).device.type
            except StopIteration:
                device = "cpu"
        self.device = device

    def accuracy(
        self,
        prompts: list[str],
        labels: list[int],
        max_length: int = 64,
    ) -> float:
        """Compute next-token prediction accuracy.

        For each prompt, the model predicts the most probable next token. The
        prediction is compared against the expected label index (position in
        the vocabulary).

        Args:
            prompts: List of input prompts.
            labels: List of expected next-token vocabulary indices.
            max_length: Maximum tokenized input length.

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for prompt, label in zip(prompts, labels):
                tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                try:
                    outputs = self.model(**tokens)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    pred = logits[0, -1].argmax().item()
                    if pred == label:
                        correct += 1
                    total += 1
                except Exception:
                    continue

        return correct / max(total, 1)

    def multiple_choice_accuracy(
        self,
        contexts: list[str],
        choices_list: list[list[str]],
        correct_indices: list[int],
    ) -> float:
        """Score multiple-choice tasks by selecting the highest-probability completion.

        For each context, the model scores each candidate choice by its log-probability
        under the model, and the choice with the highest score is selected.

        Args:
            contexts: Input context strings.
            choices_list: List of candidate completions for each context.
            correct_indices: Index of the correct choice for each context.

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for context, choices, correct_idx in zip(contexts, choices_list, correct_indices):
                scores = []
                for choice in choices:
                    full_text = context + choice
                    tokens = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                    )
                    try:
                        outputs = self.model(**tokens, labels=tokens["input_ids"])
                        scores.append(-outputs.loss.item())
                    except Exception:
                        scores.append(float("-inf"))

                if scores and max(scores) != float("-inf"):
                    pred = scores.index(max(scores))
                    if pred == correct_idx:
                        correct += 1
                    total += 1

        return correct / max(total, 1)

    def evaluate_with_metric(
        self,
        inputs: list[str],
        metric_fn: Callable[[str, str], float],
        references: list[str],
        max_new_tokens: int = 50,
    ) -> float:
        """Evaluate using a custom metric function on generated outputs.

        Args:
            inputs: Input prompts to generate from.
            metric_fn: Function that takes (generated, reference) and returns a score.
            references: Reference strings for each input.
            max_new_tokens: Number of tokens to generate.

        Returns:
            Mean metric score across all inputs.
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for prompt, reference in zip(inputs, references):
                tokens = self.tokenizer(prompt, return_tensors="pt")
                try:
                    output_ids = self.model.generate(
                        **tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                    generated = self.tokenizer.decode(
                        output_ids[0][tokens["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    scores.append(metric_fn(generated, reference))
                except Exception:
                    continue

        return sum(scores) / max(len(scores), 1)
