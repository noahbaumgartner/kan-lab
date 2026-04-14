from .base import BaseKANModel
from src.modules.wavkan.kan import KAN


class WavKANModel(BaseKANModel):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat', **kwargs):
        self.layers_hidden = layers_hidden
        self.wavelet_type = wavelet_type

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            wavelet_type=self.wavelet_type,
        ).to(device)
        self.device = device
