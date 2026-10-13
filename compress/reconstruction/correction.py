"""Small calibration fits which update only explicitly selected parameters."""
import torch
from torch import nn


class PostCompressionCorrection:
    def __init__(self, model: nn.Module, original_model: nn.Module):
        self.model, self.original = model, original_model

    @staticmethod
    def _logits(output):
        if isinstance(output, torch.Tensor):
            return output
        return output.logits if hasattr(output, 'logits') else output[0]

    def _correct(self, inputs, select, lr, steps, verbose=False):
        if not inputs or steps < 1 or lr <= 0:
            raise ValueError("Nonempty calibration data, positive steps and learning rate required")
        selected = [p for name, p in self.model.named_parameters() if select(name)]
        if not selected:
            return []
        selected_ids = {id(p) for p in selected}
        requires_grad = {p: p.requires_grad for p in self.model.parameters()}
        modes = (self.model.training, self.original.training)
        optimizer = torch.optim.Adam(selected, lr=lr)
        losses = []
        try:
            # Evaluation mode disables dropout while autograd remains enabled.
            self.model.eval()
            self.original.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(id(parameter) in selected_ids)
            for step in range(steps):
                for x in inputs:
                    with torch.no_grad():
                        target = self.original(**x) if isinstance(x, dict) else self.original(x)
                    prediction = self.model(**x) if isinstance(x, dict) else self.model(x)
                    loss = (self._logits(prediction) - self._logits(target)).square().mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite reconstruction loss")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                if verbose and step % 10 == 0:
                    print(f"step={step} loss={losses[-1]:.6g}")
        finally:
            for parameter, flag in requires_grad.items():
                parameter.requires_grad_(flag)
            self.model.train(modes[0])
            self.original.train(modes[1])
        return losses

    def correct_bias(self, calibration_inputs, lr=1e-4, steps=100, verbose=False):
        return self._correct(calibration_inputs, lambda name: name.endswith('bias'), lr, steps, verbose)

    def layer_norm_correction(self, calibration_inputs, lr=1e-3, steps=50):
        prefixes = [name for name, module in self.model.named_modules() if isinstance(module, nn.LayerNorm)]
        names = {f'{prefix}.{leaf}' if prefix else leaf for prefix in prefixes for leaf in ('weight', 'bias')}
        return self._correct(calibration_inputs, lambda name: name in names, lr, steps)
