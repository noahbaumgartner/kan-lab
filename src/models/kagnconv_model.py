import abc
import importlib
import os
import sys

import torch
import torch.nn as nn

from .base import BaseKANModel

_KAGNCONV_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "modules", "kagnconv"
)
_KAGNCONV_KANS_DIR = os.path.join(_KAGNCONV_DIR, "kans")
for _p in (_KAGNCONV_DIR, _KAGNCONV_KANS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# `kans/kan.py` and `conv_kagn_baseline.py` import `L1` from `kans.utils`,
# but the upstream torch-conv-kan repo defines it in `utils/regularization.py`,
# which isn't bundled here. Inject a faithful port into the `utils` module before
# any downstream import resolves.
class _WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for n, param in self.module.named_parameters():
                if self.name in n and (param.grad is None or torch.all(param.grad == 0.0)):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L1(_WeightDecay):
    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)


# `kans/utils.py` ends up loaded under two distinct module names: `utils`
# (because `kans/` is on sys.path) and `kans.utils` (because `kans` is also a
# namespace package). Inject L1 into both.
for _modname in ("utils", "kans.utils"):
    _m = importlib.import_module(_modname)
    _m.L1 = L1

from src.modules.kagnconv.conv_kagn_baseline import (
    SimpleConvKAGN,
    EightSimpleConvKAGN,
)


class _BaseKAGNConvModel(BaseKANModel):
    _net_cls = None
    _default_layer_sizes: tuple = ()

    def __init__(
        self,
        num_classes,
        input_channels=1,
        img_size=28,
        layer_sizes=None,
        degree=3,
        degree_out=3,
        groups=1,
        dropout=0.0,
        dropout_linear=0.0,
        l1_penalty=0.0,
        affine=True,
        **kwargs,
    ):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size
        self.layer_sizes = list(layer_sizes) if layer_sizes is not None else list(self._default_layer_sizes)
        self.degree = degree
        self.degree_out = degree_out
        self.groups = groups
        self.dropout = dropout
        self.dropout_linear = dropout_linear
        self.l1_penalty = l1_penalty
        self.affine = affine

    def build(self, device="cpu"):
        self.model = self._net_cls(
            layer_sizes=self.layer_sizes,
            num_classes=self.num_classes,
            input_channels=self.input_channels,
            degree=self.degree,
            degree_out=self.degree_out,
            groups=self.groups,
            dropout=self.dropout,
            dropout_linear=self.dropout_linear,
            l1_penalty=self.l1_penalty,
            affine=self.affine,
        ).to(device)
        self.device = device

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, self.input_channels, self.img_size, self.img_size)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.model(x)


class KAGNConv4Model(_BaseKAGNConvModel):
    """4-layer convolutional KAGN baseline (channels: 32, 64, 128, 512)."""

    _net_cls = SimpleConvKAGN
    _default_layer_sizes = (32, 64, 128, 512)


class KAGNConv8Model(_BaseKAGNConvModel):
    """8-layer convolutional KAGN baseline (channels: 32, 64, 128, 512, 1024, 1024, 1024, 1024)."""

    _net_cls = EightSimpleConvKAGN
    _default_layer_sizes = (32, 64, 128, 512, 1024, 1024, 1024, 1024)
