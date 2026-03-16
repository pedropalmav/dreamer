import torch
import torch.nn as nn
from typing import Callable


class BlockLinear(nn.Module):
    bias: bool = True
    weight_init: str | Callable = nn.init.trunc_normal_  # TODO: This is not working
    bias_init: str | Callable = nn.init.zeros_  # TODO: This is not working
    outscale: float = 1.0  # TODO: What's this for?

    def __init__(self, in_features: int, units: int, blocks: int, **kwargs):
        super().__init__()
        assert blocks <= units and units % blocks == 0
        assert in_features % blocks == 0

        self.units = units
        self.blocks = blocks
        self.in_features = in_features

        block_in = in_features // blocks
        block_out = units // blocks

        # Equivalente al kernel (blocks, block_in, block_out)
        self.weight = nn.Parameter(
            self._scaled_weight_init(torch.empty(blocks, block_in, block_out))
        )

        if self.bias:
            self.bias_param = (
                nn.Parameter(nn.init.zeros_(torch.empty(units))) if self.bias else None
            )

    # TODO: Check this implementation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        block_in = self.in_features // self.blocks

        # (..., blocks, block_in)
        x = x.reshape(*x.shape[:-1], self.blocks, block_in)

        # (..., blocks, block_out)
        x = torch.einsum("...ki,kio->...ko", x, self.weight)

        # (..., units)
        x = x.reshape(*x.shape[:-2], self.units)

        if self.bias_param is not None:
            x = x + self.bias_param

        return x

    def _scaled_weight_init(self, tensor):
        return nn.init.trunc_normal_(tensor) * self.outscale
