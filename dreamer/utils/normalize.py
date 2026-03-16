import torch
import torch.nn as nn


# TODO: Complete
class Normalize(nn.Module):

    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = True

    def __init__(self, impl, **kwargs):
        super().__init__()
        self.impl = impl

        if self.debias and self.impl != "none":
            self.corr = None  # nj.Variable

        if self.impl == "none":
            pass
        elif self.impl == "meanstd":
            self.mean = None  # nj.Variable
            self.std = None  # nj.Variable
        elif self.impl == "perc":
            self.lo = None  # nj.Variable
            self.hi = None  # nj.Variable
        else:
            raise NotImplementedError(self.impl)
