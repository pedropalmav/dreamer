import math
import torch
import torch.nn as nn
import numpy as np
from dreamer.heads.dict_head import DictHead
from dreamer.networks import BlockLinear
from dreamer.networks import MLP
import utils.jax_nets as jn
import utils.jnp as jnp
import dreamer.outs as outs
import einops


class Decoder(nn.Module):

    units: int = 1024
    normalization: str = "rms"
    activation: str = "gelu"
    outscale: float = 1.0
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    bspace: int = 8
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kwargs):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        super(Decoder, self).__init__()
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        self.kwargs = kwargs

        if self.veckeys:
            mlp_kw = self.kwargs.copy()
            mlp_kw["act"] = self.activation
            mlp_kw["norm"] = self.normalization
            head_kw = self.kwargs.copy()
            head_kw["outscale"] = self.outscale

            spaces = {k: self.obs_space[k] for k in self.veckeys}
            o1, o2 = "categorical", ("symlog_mse" if self.symlog else "mse")
            outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}

            mlp_kw.pop("layers")
            # TODO: Calculate input size
            self.mlp = MLP(
                layers=self.layers, input_size=1, hidden_size=self.units, **mlp_kw
            )
            self.head = DictHead(self.units, spaces, outputs, **head_kw)

        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            self.minres = [int(x // factor) for x in self.imgres]
            assert 3 <= self.minres[0] <= 16, self.minres
            assert 3 <= self.minres[1] <= 16, self.minres
            shape = (*self.minres, self.depths[-1])

            if self.bspace:
                u, g = math.prod(shape), self.bspace
                kw = self.kwargs.copy()
                kw.pop("units")
                self.sp0 = BlockLinear(
                    self.bspace, u, g, **kw
                )  # TODO: Calculate input size

                self.sp1 = nn.Linear(u, 2 * self.units)  # TODO: Calculate input size
                self.sp1norm = nn.RMSNorm(2 * self.units)
                self.sp1act = nn.GELU()

                self.sp2 = nn.Linear(u, 1)  # TODO: Calculate input size, fix shape
                self.spnorm = nn.RMSNorm(1)
                self.spact = nn.GELU()

            else:
                self.space = MLP(
                    layers=1,
                    input_size=None,  # TODO: Calculate input size
                    hidden_size=1,  # TODO: Fix shape
                    **self.kwargs,
                )

            # TODO: Refactor this CNN, and adapt to allow transposed convolutions
            cnn = {}
            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    # kw = dict(**self.kwargs, transp=True)
                    # TODO: Calculate input channels
                    cnn[f"conv{i}"] = nn.Conv2d(3, depth, self.kernel, stride=2)
                else:
                    # TODO: Calculate input channels, Add kwargs
                    cnn[f"conv{i}"] = nn.Conv2d(3, depth, self.kernel, stride=1)

                cnn[f"conv{i}norm"] = nn.RMSNorm(depth)
                cnn[f"conv{i}act"] = nn.GELU()

            self.cnn = nn.ModuleDict(cnn)

            # TODO: Calculate input channels, Add kwargs
            if self.outer:
                # kw = dict(**self.kwargs, outscale=self.outscale)
                self.imgout = nn.Conv2d(3, self.imgdep, self.kernel)
            elif self.strided:
                # kw = dict(**self.kwargs, outscale=self.outscale, transp=True)
                self.imgout = nn.Conv2d(3, self.imgdep, self.kernel)
            else:
                # kw = dict(**self.kwargs, outscale=self.outscale)
                self.imgout = nn.Conv2d(3, self.imgdep, self.kernel)

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def forward(self, carry, feat, reset, training, single=False):
        assert feat["deterministic"].shape[-1] % self.bspace == 0
        K = self.kernel
        recons = {}
        bshape = reset.shape
        inp = [jn.cast(feat[k]) for k in ("stochastic", "deterministic")]
        inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
        inp = torch.cat(inp, dim=-1)

        if self.veckeys:
            x = self.mlp(inp)
            x = x.reshape((*bshape, *x.shape[1:]))
            outs = self.head(x)
            recons.update(outs)

        if self.imgkeys:
            if self.bspace:
                # Reshaping
                x0, x1 = jn.cast((feat["deterministic"], feat["stochastic"]))
                x1 = x1.reshape((*x1.shape[:-2], -1))
                x0 = x0.reshape((-1, x0.shape[-1]))
                x1 = x1.reshape((-1, x1.shape[-1]))

                x0 = self.sp0(x0)
                x0 = einops.rearrange(
                    x0,
                    "... (g h w c) -> ... hw (g c)",
                    h=self.minres[0],
                    w=self.minres[1],
                    g=self.bspace,
                )
                x1 = self.sp1(x1)
                x1 = self.sp1act(self.sp1norm(x1))
                x1 = self.sp2(x1)
                x = self.spact(self.spnorm(x0 + x1))

            else:
                x = self.space(inp)

            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                x = self.cnn[f"conv{i}"](x)
                x = self.cnn[f"conv{i}norm"](x)
                x = self.cnn[f"conv{i}act"](x)

            if self.outer or self.strided:
                x = self.imgout(x)
            else:
                x = x.repeat(2, -2).repeat(2, -3)
                x = self.imgout(x)

            x = nn.Sigmoid(x)
            x = x.reshape((*bshape, *x.shape[1:]))
            split = np.cumsum([self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])

            for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
                out = outs.MSE(out)
                out = None  # TODO: Implement Agg
                recons[k] = out

        entries = {}
        return carry, entries, recons
