import torch
import torch.nn as nn
from typing import Optional


class PostCompressionCorrection:
    """Recover accuracy lost to quantization via a lightweight calibration pass.

    After aggressive quantization, output logits can drift significantly from
    the original model. This class minimizes the mean-squared error between
    the compressed model's outputs and the original model's outputs on a small
    calibration set by fine-tuning only the bias parameters — the lightest
    possible adaptation that can partially compensate for systematic quantization
    error.

    For stronger correction, consider full bias + LayerNorm fine-tuning, or
    GPTQ-style layer-wise reconstruction.

    Args:
        model: The compressed model to correct.
        original_model: The original (uncompressed) teacher model.
    """

    def __init__(self, model: nn.Module, original_model: nn.Module):
        self.model = model
        self.original = original_model

    def correct_bias(
        self,
        calibration_inputs: list[torch.Tensor],
        lr: float = 1e-4,
        steps: int = 100,
        verbose: bool = False,
    ) -> list[float]:
        """Minimize output difference on calibration data by adjusting biases.

        Freezes all parameters except biases, then runs Adam optimization to
        minimize the MSE between compressed and original model logits.

        Args:
            calibration_inputs: List of tokenized input dicts (as returned by
                a HuggingFace tokenizer with return_tensors="pt").
            lr: Learning rate for the Adam optimizer.
            steps: Number of optimization steps.
            verbose: If True, print loss every 10 steps.

        Returns:
            List of per-step loss values.
        """
        self.model.train()
        params = [p for n, p in self.model.named_parameters() if "bias" in n]
        if not params:
            return []

        optimizer = torch.optim.Adam(params, lr=lr)
        losses = []
        batch = calibration_inputs[:10]

        for step in range(steps):
            for x in batch:
                with torch.no_grad():
                    target = self.original(**x) if isinstance(x, dict) else self.original(x)
                pred = self.model(**x) if isinstance(x, dict) else self.model(x)

                t_logits = target.logits if hasattr(target, "logits") else target[0]
                p_logits = pred.logits if hasattr(pred, "logits") else pred[0]

                loss = ((p_logits - t_logits) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if verbose and step % 10 == 0:
                avg = sum(losses[-len(batch):]) / max(len(batch), 1)
                print(f"  step {step:4d}/{steps}  avg_loss={avg:.6f}")

        self.model.eval()
        return losses

    def layer_norm_correction(
        self,
        calibration_inputs: list[torch.Tensor],
        lr: float = 1e-3,
        steps: int = 50,
    ) -> list[float]:
        """Fine-tune LayerNorm scale and shift parameters to correct output drift.

        A lightweight complement to correct_bias that adjusts normalization
        layer statistics to compensate for distribution shifts introduced
        by quantization.

        Args:
            calibration_inputs: List of tokenized input dicts.
            lr: Learning rate.
            steps: Number of optimization steps.

        Returns:
            List of per-step loss values.
        """
        self.model.train()
        params = [
            p for n, p in self.model.named_parameters()
            if any(k in n for k in ["ln", "layer_norm", "layernorm", "norm"])
        ]
        if not params:
            return []

        optimizer = torch.optim.Adam(params, lr=lr)
        losses = []
        batch = calibration_inputs[:10]

        for step in range(steps):
            for x in batch:
                with torch.no_grad():
                    target = self.original(**x) if isinstance(x, dict) else self.original(x)
                pred = self.model(**x) if isinstance(x, dict) else self.model(x)

                t_logits = target.logits if hasattr(target, "logits") else target[0]
                p_logits = pred.logits if hasattr(pred, "logits") else pred[0]

                loss = ((p_logits - t_logits) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        self.model.eval()
        return losses
