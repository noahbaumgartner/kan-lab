import sys
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
