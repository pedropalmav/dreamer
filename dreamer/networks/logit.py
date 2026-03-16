import torch
import torch.nn as nn


class Logit(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(Logit, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
