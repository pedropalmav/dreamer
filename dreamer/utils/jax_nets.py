import torch
import torch.utils._pytree as pytree
import torch.nn.functional as F
import numpy as np

COMPUTE_DTYPE = torch.float


def cast(xs, force=False):
    """
    Convierte tensores flotantes a COMPUTE_DTYPE (bfloat16),
    preservando enteros y booleanos a menos que force=True.
    """

    if force:
        should_cast = lambda x: True
    else:
        should_cast = lambda x: x.is_floating_point()

    return pytree.tree_map(lambda x: x.to(COMPUTE_DTYPE) if should_cast(x) else x, xs)


def mask(xs, mask):
    return where(mask, xs, pytree.tree_map(torch.zeros_like, xs))


def where(condition, xs, ys):
    assert condition.dtype == torch.bool, condition.dtype

    def fn(x, y):
        assert x.shape == y.shape, (x.shape, y.shape)
        expanded = condition.view(condition.shape + (1,) * (x.ndim - condition.ndim))
        return torch.where(expanded, x, y)

    return pytree.tree_map(fn, xs, ys)


def available(*trees, bdims=None):
    def fn(*xs):
        masks = []
        for x in xs:
            if torch.is_floating_point(x):
                mask = x != float("-inf")
            elif is_signed_integer(x.dtype):
                mask = x != -1
            elif torch.is_unsigned(x.dtype) or x.dtype == torch.bool:
                shape = x.shape if bdims is None else x.shape[:bdims]
                mask = torch.full(shape, True, dtype=torch.bool, device=x.device)
            else:
                raise NotImplementedError(x.dtype)

            if bdims is not None:
                mask = mask.all(tuple(range(bdims, mask.ndim)))
            masks.append(mask)
        return torch.stack(masks, 0).all(0)

    return pytree.tree_map(fn, *trees)


def is_signed_integer(dtype):
    return dtype in {torch.int8, torch.int16, torch.int32, torch.int64}


class DictConcat:
    def __init__(self, spaces, fdims, squish=lambda x: x):
        assert 1 <= fdims
        self.keys = sorted(spaces.keys())
        self.spaces = spaces
        self.fdims = fdims  # Feature dims
        self.squish = squish

    def __call__(self, xs):
        assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())

        # Calcular batch dims
        first_key = self.keys[0]
        bdims = xs[first_key].ndim - len(self.spaces[first_key].shape)

        # TODO: Create method for this loop
        ys = []
        for key in self.keys:
            space = self.spaces[key]
            x = xs[key]
            m = available(x, bdims=bdims)
            x = mask(x, m)

            assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)

            # TODO: Create method for this part
            if space.dtype == torch.uint8 and len(space.shape) in (2, 3):
                raise NotImplementedError("Images are not supported")
            elif space.discrete:
                classes = torch.as_tensor(space.classes).flatten()
                assert (classes == classes[0]).all(), (key, classes)
                classes = classes[0].item()
                x = x.long()
                x = F.one_hot(x, classes).to(COMPUTE_DTYPE)
            else:
                x = self.squish(x)
                x = x.to(COMPUTE_DTYPE)

            x = mask(x, m)
            x = x.reshape((*x.shape[: bdims + self.fdims - 1], -1))
            ys.append(x)

        return torch.cat(ys, -1)
