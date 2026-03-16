import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from typing import Callable
from dreamer.networks import MLP
from dreamer.heads import Head, DictHead


class MLPHead(nn.Module):
    units: int = 1024
    layers: int = 5
    act: str = "silu"
    norm: str = "rms"
    bias: bool = True
    winit: str | Callable = nn.init.trunc_normal_
    binit: str | Callable = nn.init.zeros_

    def __init__(self, input_size, space, output, **hkw):
        super().__init__()
        shared = dict(bias=self.bias, winit=self.winit, binit=self.binit)
        mkw = dict(**shared, act=self.act, norm=self.norm)  # MLP kwargs
        # hkw = dict(**shared, **hkw)  # TODO: Head kwargs

        self.mlp = MLP(self.layers, input_size, self.units, **mkw)

        if isinstance(space, dict):
            self.head = DictHead(self.units, space, output, **hkw)
        else:
            self.head = Head(self.units, space, output, **hkw)

    def forward(self, x, bdims):
        bshape = pytree.tree_leaves(x)[0].shape[:bdims]
        x = x.reshape((*bshape, -1))
        x = self.mlp(x)
        x = self.head(x)
        return x
