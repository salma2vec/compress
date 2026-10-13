"""Fail-visible task metrics with explicit completion-token scoring."""
import math
from contextlib import contextmanager

import torch


class TaskEvaluator:
    def __init__(self, model, tokenizer, device=None):
        self.model, self.tokenizer = model, tokenizer
        self.device = torch.device(device) if device else next(model.parameters(), next(model.buffers(), torch.empty(0))).device

    @contextmanager
    def _evaluation(self):
        training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.model.train(training)

    def _tokens(self, text, **kwargs):
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in self.tokenizer(text, return_tensors="pt", **kwargs).items()}

    @staticmethod
    def _validate(*items):
        if not items[0] or len({len(item) for item in items}) != 1:
            raise ValueError("Inputs must be nonempty and have equal lengths")

    def accuracy(self, prompts, labels, max_length=64):
        """Next-token vocabulary accuracy; failed examples stop the evaluation."""
        self._validate(prompts, labels)
        correct = 0
        with self._evaluation():
            for prompt, label in zip(prompts, labels):
                outputs = self.model(**self._tokens(prompt, truncation=True, max_length=max_length))
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite task logits")
                correct += int(logits[0, -1].argmax() == label)
        return correct / len(prompts)

    def multiple_choice_accuracy(self, contexts, choices_list, correct_indices):
        """Select by summed completion log likelihood; exclude context loss.

        Requires a fast tokenizer with character offsets. Full tokenization avoids
        assuming that separately tokenizing context preserves BPE boundaries.
        Tokens straddling the context boundary count as completion tokens.
        """
        self._validate(contexts, choices_list, correct_indices)
        correct = 0
        with self._evaluation():
            for context, choices, correct_idx in zip(contexts, choices_list, correct_indices):
                if not context or not choices or not 0 <= correct_idx < len(choices):
                    raise ValueError("Context, choices and valid correct index required")
                scores = []
                for choice in choices:
                    if not choice:
                        raise ValueError("Choices must be nonempty")
                    tokens = self._tokens(context + choice, truncation=False, return_offsets_mapping=True)
                    offsets = tokens.pop('offset_mapping')
                    labels = tokens['input_ids'].clone()
                    labels[offsets[..., 1] <= len(context)] = -100
                    count = int((labels[:, 1:] != -100).sum())
                    if not count:
                        raise ValueError("Choice has no predictable tokens")
                    loss = self.model(**tokens, labels=labels).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite choice likelihood")
                    scores.append(-float(loss) * count)
                correct += scores.index(max(scores)) == correct_idx
        return correct / len(contexts)

    def evaluate_with_metric(self, inputs, metric_fn, references, max_new_tokens=50):
        self._validate(inputs, references)
        scores = []
        with self._evaluation():
            for prompt, reference in zip(inputs, references):
                tokens = self._tokens(prompt)
                output = self.model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=False)
                generated = self.tokenizer.decode(output[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
                score = float(metric_fn(generated, reference))
                if not math.isfinite(score):
                    raise ValueError("Nonfinite metric")
                scores.append(score)
        return sum(scores) / len(scores)
