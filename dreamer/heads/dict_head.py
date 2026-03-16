import torch
import torch.nn as nn

from dreamer.heads.head import Head


class DictHead(nn.Module):

    def __init__(self, input_size, spaces, outputs, **kwargs):
        super().__init__()

        assert spaces, spaces
        if not isinstance(spaces, dict):
            spaces = {"output": spaces}
        if not isinstance(outputs, dict):
            outputs = {"output": outputs}
        assert spaces.keys() == outputs.keys(), (spaces, outputs)

        self.kwargs = kwargs

        self.heads = nn.ModuleDict(
            {
                key: Head(input_size, spaces[key], impl, **kwargs)
                for key, impl in outputs.items()
            }
        )

    def forward(self, x):
        return {key: head(x) for key, head in self.heads.items()}
