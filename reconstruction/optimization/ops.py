import torch
from mei.legacy.utils import varargin


class ChangeNormConditional:
    """ Change the norm of the input.

    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm):
        self.norm = norm

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        if x_norm >= self.norm:
            x = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        return x

class ChangeNormAndClip:
    """ Change the norm of the input.
    Arguments:
        norm (float or tensor): Desired norm. If tensor, it should be the same length as
            x.
    """

    def __init__(self, norm, x_min, x_max):
        self.norm = norm
        self.x_min = x_min
        self.x_max = x_max

    @varargin
    def __call__(self, x, iteration=None):
        x_norm = torch.norm(x.view(len(x), -1), dim=-1)
        if x_norm >= self.norm:
            x = x * (self.norm / x_norm).view(len(x), *[1] * (x.dim() - 1))
        
        assert len(self.x_min) == len(self.x_max) == x.shape[1]
        for c in range(x.shape[1]):
            x[0,:][c] = torch.clamp(x[0,:][c], self.x_min[c], self.x_max[c])
        return x