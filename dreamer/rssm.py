import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from dreamer.networks.logit import Logit
from dreamer.networks.mlp import MLP
from dreamer.networks.block_linear import BlockLinear
from dreamer.outs import OneHot
import utils.ninjax as nj
import utils.jax_nets as jn
import utils.jnp as jnp
import einops
import numpy as np


def sg(x):
    return x.detach()


class RSSM(nn.Module):

    deterministic_size: int = 4096
    hidden_size: int = 2048
    stochastic_size: int = 32
    classes: int = 32
    normalization: str = "rms"
    activation: str = "gelu"
    unroll: bool = False
    uniform_mix: float = 0.01
    outscale: float = 1.0
    img_layers: int = 2
    obs_layers: int = 1
    dynamic_layers: int = 1
    absolute: bool = False
    blocks: int = 8
    free_nats: float = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, action_space, **kwargs):
        super(RSSM, self).__init__()
        self.action_space = action_space
        self.kwargs = kwargs

        # Networks
        self.prior = MLP(
            layers=self.img_layers,
            input_size=self.deterministic_size,
            hidden_size=self.hidden_size,
            **kwargs,
        )

        obs_net_input = (
            5120 if self.absolute else 5120 + self.deterministic_size
        )  # TODO: Fix this
        # TODO: Find the name for this
        self.obs_net = MLP(
            layers=self.obs_layers,
            input_size=obs_net_input,
            hidden_size=self.hidden_size,
            **kwargs,
        )

        self.dynin_0 = MLP(
            layers=1,
            input_size=self.deterministic_size,
            hidden_size=self.hidden_size,
            **kwargs,
        )

        self.dynin_1 = MLP(
            layers=1,
            input_size=self.stochastic_size * self.classes,
            hidden_size=self.hidden_size,
            **kwargs,
        )

        self.dynin_2 = MLP(
            layers=1,
            input_size=self._get_action_input_size(),
            hidden_size=self.hidden_size,
            **kwargs,
        )

        # TODO: Check this
        kwargs.pop("blocks")
        block_in_size = 3 * self.hidden_size * self.blocks + self.deterministic_size
        self.dynlayers = nn.Sequential(
            *[
                BlockLinear(
                    in_features=block_in_size,
                    units=self.deterministic_size,
                    blocks=self.blocks,
                    **kwargs,
                ),
                nn.RMSNorm(self.deterministic_size),  # TODO: Check this
                nn.GELU(),
            ]  # TODO: Add more layers based on self.dynamic_layers
        )

        self.dyngru = BlockLinear(
            in_features=self.deterministic_size,  # TODO: Calculate this
            units=3 * self.deterministic_size,
            blocks=self.blocks,
            **kwargs,
        )

        logits_kwargs = kwargs.copy()
        logits_kwargs["outscale"] = self.outscale
        self.logits = nn.ModuleDict(
            {
                "priorlogit": Logit(
                    input_size=self.hidden_size,
                    output_size=self.stochastic_size * self.classes,
                    **logits_kwargs,
                ),
                "obslogit": Logit(
                    input_size=self.hidden_size,
                    output_size=self.stochastic_size * self.classes,
                    **logits_kwargs,
                ),
            }
        )

    def _get_action_input_size(self):
        size = 0
        for key, space in self.action_space.items():
            if space.discrete:
                classes = np.asarray(space.classes).flatten()
                classes = classes[0].item()
                size += np.prod(space.shape, dtype=int) * classes
            else:
                size += np.prod(space.shape, dtype=int)
        return size

    # TODO: Rename
    def initial(self, batch_size):
        carry = dict(
            deterministic=torch.zeros(
                batch_size,
                self.deterministic_size,
                dtype=torch.float32,
                device=self.device,
            ),
            stochastic=torch.zeros(
                batch_size,
                self.stochastic_size,
                self.classes,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return carry

    # TODO: Refactor to give more context
    def truncate(self, entries, carry=None):
        """
        tree_map lo que hace es aplciarle una función a todas las hojas de estructura de datos.
        En este caso, entries es un diccionario.
        """
        assert entries["deterministic"].ndim == 3, entries["deterministic"].shape
        carry = pytree.tree_map(lambda x: x[:, -1], entries)
        return carry

    # TODO: Refactor to give more context
    def starts(self, entries, carry, n_last):
        first_leaf = pytree.tree_leaves(carry)[0]
        batch_size = first_leaf.shape[0]
        return pytree.tree_map(
            lambda x: x[:, -n_last:].reshape(batch_size * n_last, *x.shape[2:]),
            entries,
        )

    # TODO: Refactor to give more context
    def observe(self, carry, tokens, action, reset, training, single=False):
        carry, tokens, action = jn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
            return carry, entry, feat
        else:
            unroll = pytree.tree_leaves(tokens)[0].shape[1] if self.unroll else 1
            carry, (entries, feat) = nj.scan(
                lambda carry, inputs: self._observe(carry, *inputs, training),
                carry,
                (tokens, action, reset),
                unroll=unroll,
                axis=1,
            )

            return carry, entries, feat

    def _observe(self, carry, tokens, action, reset, training):
        # Fase 1: Gestión de fronteras de episodios
        deter, stoch, action = jn.mask(
            (carry["deterministic"], carry["stochastic"], action), ~reset
        )

        action = jn.DictConcat(self.action_space, 1)(action)
        action = jn.mask(action, ~reset)

        # Fase 2: Paso determinista
        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else torch.cat([deter, tokens], dim=-1)

        ## Fase 3: Paso estocástico (posterior network)
        x = self.obs_net(x)
        logit = self._logit("obslogit", x)
        stoch = jn.cast(self._dist(logit).sample())

        ## Fase 4: Parse data
        carry = dict(deterministic=deter, stochastic=stoch)
        entry = dict(deterministic=deter, stochastic=stoch, logit=logit)
        feat = dict(deterministic=deter, stochastic=stoch)
        assert all(x.dtype == jn.COMPUTE_DTYPE for x in (deter, stoch, logit))
        return carry, (entry, feat)

    def imagine(self, carry, policy, length, training, single=False):
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            actemb = None  # TODO: nn.DictConcat
            deter = self._core(carry["deterministic"], carry["stochastic"], actemb)
            logit = self._prior(deter)
            stoch = jn.cast(self._dist(logit).sample(seed=None))  # TODO: Add real seed

            # Parse data
            carry = jn.cast(dict(deterministic=deter, stochastic=stoch))
            feat = jn.cast(dict(deterministic=deter, stochastic=stoch, logit=logit))
            assert all(x.dtype == jn.COMPUTE_DTYPE for x in (deter, stoch, logit))
            return carry, (feat, action)
        else:
            unroll = length if self.unroll else 1
            if callable(policy):
                carry, (feat, action) = nj.scan(
                    lambda c, _: self.imagine(c, policy, 1, training, single=True),
                    jn.cast(carry),
                    (),
                    length,
                    unroll=unroll,
                    axis=1,
                )
            else:
                carry, (feat, action) = nj.scan(
                    lambda c, a: self.imagine(c, a, 1, training, single=True),
                    jn.cast(carry),
                    jn.cast(policy),
                    length,
                    unroll=unroll,
                    axis=1,
                )

            return carry, (feat, action)

    def loss(self, carry, tokens, acts, reset, training):
        metrics = {}
        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
        prior = self._prior(feat["deterministic"])
        post = feat["logit"]
        dyn = self._dist(sg(post)).kl(self._dist(prior))  # TODO: dynamics_loss
        rep = self._dist(post).kl(sg(self._dist(prior)))  # TODO: representation_loss
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        losses = {"dyn": dyn, "rep": rep}
        metrics["dyn_ent"] = self._dist(prior).entropy().mean()
        metrics["rep_ent"] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics

    # RNN
    def _core(self, deter, stoch, action):
        # Aplana todas las dimensiones excepto la de batch
        stoch = stoch.reshape(stoch.shape[0], -1)
        action /= sg(torch.clamp(jnp.abs(action), min=1))  # np.maximum fue reemplazado
        g = self.blocks  # TODO: Group
        # Transforma un vector de tamaño (batch, features) a (batch, groups, features_per_group)
        flat2group = lambda x: einops.rearrange(x, "... (g h) -> ... g h", g=g)
        # Transforma un vector de tamaño (batch, groups, features_per_group) a (batch, features)
        group2flat = lambda x: einops.rearrange(x, "... g h -> ... (g h)", g=g)

        x0 = self.dynin_0(deter)
        x1 = self.dynin_1(stoch)
        x2 = self.dynin_2(action)

        x = torch.cat([x0, x1, x2], dim=-1)[..., None, :]
        # torch.repeat_interleave es el equivalente a np.repeat
        x = torch.repeat_interleave(x, g, dim=-2)

        x = group2flat(torch.cat([flat2group(deter), x], dim=-1))
        x = self.dynlayers(x)

        x = self.dyngru(x)
        gates = torch.chunk(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(x) for x in gates]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        deter = update * deter + (1 - update) * cand
        return deter

    def _prior(self, feat):
        x = feat
        x = self.prior(x)
        return self._logit("priorlogit", x)

    def _logit(self, name, x):
        x = self.logits[name](x)
        return x.reshape(x.shape[:-1] + (self.stochastic_size, self.classes))

    def _dist(self, logits):
        return OneHot(logits=logits, unimix=self.uniform_mix)
