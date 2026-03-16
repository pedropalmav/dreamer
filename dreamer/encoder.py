import torch
import torch.nn as nn
import numpy as np
import utils.jax_nets as jn
from networks.mlp import MLP
from networks.cnn import CNN


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


class Encoder(nn.Module):

    units: int = 1024
    normalization: str = "rms"
    activation: str = "gelu"
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kwargs):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        super(Encoder, self).__init__()
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kwargs = kwargs

        if self.veckeys:
            kwargs.pop("layers")
            self.mlp = MLP(
                layers=self.layers,
                input_size=self._get_veckeys_input_size(),
                hidden_size=self.units,
                **kwargs,
            )

        if self.imgkeys:
            stride = 2 if self.strided else 1
            self.cnn = CNN(
                input_channels=3 * len(self.imgkeys),  # TODO: Check if this is correct
                depths=self.depths,
                kernel_size=self.kernel,
                stride=stride,
                max_pool=True,
                **kwargs,
            )

    def _get_veckeys_input_size(self):
        size = 0
        for key in self.veckeys:
            space = self.obs_space[key]
            if space.discrete:
                classes = np.asarray(space.classes).flatten()
                classes = classes[0].item()
                size += np.prod(space.shape, dtype=int) * classes
            else:
                size += np.prod(space.shape, dtype=int)
        return size

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def forward(self, carry, obs, reset, training, single=False):
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = symlog if self.symlog else lambda x: x
            x = jn.DictConcat(vspace, 1, squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            x = self.mlp(x)
            outs.append(x)

        if self.imgkeys:
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == torch.uint8 for x in imgs)
            # Concatenar todas las imágenes y normalizarlas del rango [0, 255] a [-0.5, 0.5]
            x = jn.cast(torch.cat(imgs, -1), force=True) / 255.0 - 0.5
            # Aplana las dimensiones de batch y time para procesar las imágenes como un solo batch
            x = x.reshape((-1, *x.shape[bdims:]))
            x = x.permute(
                0, 3, 1, 2
            )  # TODO: No sé si esto debería ir dentro del modelo o fuera

            x = self.cnn(x)
            x = x.permute(0, 2, 3, 1)
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            x = x.reshape((x.shape[0], -1))
            outs.append(x)

        x = torch.cat(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens
