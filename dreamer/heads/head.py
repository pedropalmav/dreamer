import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import elements
import dreamer.outs as outs
from dreamer.utils import jnp

f32 = np.float32


class Head(nn.Module):

    minstd: float = 1.0
    maxstd: float = 1.0
    unimix: float = 0.0
    bins: int = 255
    outscale: float = 1.0

    def __init__(self, input_size, space, output, **kwargs):
        super().__init__()
        if isinstance(space, tuple):
            space = elements.Space(np.float32, space)

        if output == "onehot":
            classes = np.asarray(space.classes).flatten()
            assert (classes == classes[0]).all(), classes
            shape = (*space.shape, classes[0].item())
            space = elements.Space(f32, shape, 0.0, 1.0)

        self.space = space
        self.impl = output
        self.kwargs = kwargs

        output_shape = self._output_shape()
        self._build_layers(input_size, output_shape, self.outscale)

    def _output_shape(self):
        if self.impl == "categorical":
            assert self.space.discrete
            classes = np.asarray(self.space.classes).flatten()
            assert (classes == classes[0]).all(), classes
            return (*self.space.shape, classes[0].item())
        elif self.impl == "bounded_normal":
            assert not self.space.discrete
            return self.space.shape

    def _build_layers(self, input_size, output_shape, outscale):
        if self.impl == "categorical":
            self.logits = nn.Linear(input_size, output_shape[0])
        elif self.impl == "bounded_normal":
            self.mean = nn.Linear(input_size, output_shape[0])
            self.std = nn.Linear(input_size, output_shape[0])

    def forward(self, x):
        if not hasattr(self, self.impl):
            raise NotImplementedError(self.impl)

        # Esto invoca el método correspondiente
        output = getattr(self, self.impl)(x)
        if self.space.shape:
            # TODO: Fix this
            # output = outs.Agg(output, len(self.space.shape), jnp.sum)
            pass

        assert output.sample().shape[x.ndim - 1 :] == self.space.shape, (
            self.space,
            self.impl,
            x.shape,
            output.sample().shape,
        )
        return output

    def binary(self, x):
        assert np.all(self.space.classes == 2), self.space
        logit = None  # TODO: Forward pass in Linear layer
        return outs.Binary(logit)

    def categorical(self, x):
        logits = self.logits(x)
        output = D.Categorical(logits=logits)
        output.minent = 0
        output.maxent = np.log(logits.shape[-1])
        return output

    def onehot(self, x):
        assert not self.space.discrete
        logits = None  # TODO: Forward pass in Linear layer
        return outs.OneHot(logits)

    def mse(self, x):
        assert not self.space.discrete
        pred = None  # TODO: Forward pass in Linear layer
        return outs.MSE(pred)

    def huber(self, x):
        assert not self.space.discrete
        pred = None  # TODO: Forward pass in Linear layer
        return outs.Huber(pred)

    def symlog_mse(self, x):
        assert not self.space.discrete
        pred = None  # TODO: Forward pass in Linear layer
        return outs.SymlogMSE(pred, None)  # TODO: Move symlog implementation to utils

    def bounded_normal(self, x):
        mean = self.mean(x)
        stddev = self.std(x)
        lo, hi = self.minstd, self.maxstd
        stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
        output = D.Normal(torch.tanh(mean), stddev)
        output.minent = D.Normal(torch.zeros_like(mean), self.minstd).entropy()
        output.maxent = D.Normal(torch.zeros_like(mean), self.maxstd).entropy()
        return output

    """
    Revisando el código de Dreamer, al parecer el resto de funciones no se utilizan, por lo que no las implementaré de momento.
    """
