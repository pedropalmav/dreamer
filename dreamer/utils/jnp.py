import torch

uint8 = torch.uint8


def maximum(x, y):
    return torch.maximum(x, y)


def abs(x):
    return torch.abs(x)


def split(x, num_or_size_splits, dim=0):
    return torch.split(x, num_or_size_splits, dim)


def isfinite(x):
    return torch.isfinite(x)


def pad(x, pad_width, mode="constant", **kwargs):
    return torch.nn.functional.pad(x, pad_width, mode=mode, **kwargs)


def full(size, fill_value, *, out=None, dtype=None, device=None, requires_grad=False):
    return torch.full(
        size, fill_value, dtype=dtype, device=device, requires_grad=requires_grad
    )


def array(
    object, dtype=None, copy=True, order="K", ndmin=0, *, device=None, out_sharding=None
):
    return torch.tensor(object, dtype=dtype, device=device)


def where(condition, x=None, y=None):
    return torch.where(condition, x, y)


def concatenate(arrays, axis=0, dtype=None):
    return torch.cat(arrays, dim=axis)
