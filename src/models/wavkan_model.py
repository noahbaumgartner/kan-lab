import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules" / "wavkan"))

from .base import BaseKANModel
from modules.wavkan.KAN import KAN


class WavKANModel(BaseKANModel):
    def __init__(self, layers_hidden, wavelet_type="mexican_hat", **kwargs):
        self.layers_hidden = layers_hidden
        self.wavelet_type = wavelet_type

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            wavelet_type=self.wavelet_type,
        ).to(device)
        self.device = device

        # Disable BatchNorm (destroys scale for regression) and
        # enable linear residual path for gradient flow
        for layer in self.model.layers:
            layer.bn = torch.nn.Identity()
            layer.use_base = True

        # Patch forward to include base_output residual
        import types
        def _forward_with_residual(self, x):
            wavelet_output = self.wavelet_transform(x)
            base_output = torch.nn.functional.linear(x, self.weight1)
            combined_output = wavelet_output + base_output
            return self.bn(combined_output)

        for layer in self.model.layers:
            layer.forward = types.MethodType(_forward_with_residual, layer)
