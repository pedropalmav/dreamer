import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers: int, input_size: int, hidden_size: int, **kwargs):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            self._make_block(input_size, hidden_size),
            *[self._make_block(hidden_size, hidden_size) for _ in range(layers - 1)],
        )

    def _make_block(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features), nn.RMSNorm(out_features), nn.GELU()
        )

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)
