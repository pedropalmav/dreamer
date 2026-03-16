import torch
import torch.nn as nn
from typing import Tuple


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.RMSNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


# TODO: Check if MaxPool applies
class CNN(nn.Module):

    def __init__(
        self,
        depths: Tuple[int],
        kernel_size: int,
        input_channels: int = 3,
        stride: int = 1,
        max_pool: bool = False,
        **kwargs,
    ):
        super(CNN, self).__init__()
        self.depths = depths
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride
        self.max_pool = max_pool
        self.kwargs = kwargs

        self.cnn = nn.Sequential(
            self._make_block(self.input_channels, self.depths[0]),
            *[
                self._make_block(self.depths[i], self.depths[i + 1])
                for i in range(len(self.depths) - 1)
            ],
        )

    # TODO: Change this method to accept no pooling operation on first block if Encoder.outer is True
    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                self.kernel_size,
                stride=self.stride,
                padding="same",
            ),
            *([nn.MaxPool2d(2, 2)] if self.max_pool else []),
            RMSNorm2d(out_channels),
            nn.GELU(),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)
