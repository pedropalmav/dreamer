import torch
import torch.functional as F
from torch.distributions import OneHotCategorical


# Inspired by https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L795
class OneHot(OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix=0.0):
        if unimix > 0 and logits is not None:
            probs_mixed = torch.softmax(logits, dim=-1)
            probs_mixed = probs_mixed * (1.0 - unimix) + unimix / logits.shape[-1]
            logits = torch.log(probs_mixed)

        super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=torch.Size(), seed=None):
        if seed is not None:
            raise ValueError("Seeding is not supported in this implementation.")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample
