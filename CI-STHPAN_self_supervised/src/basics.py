
import sys

import torch

import collections
from collections import OrderedDict


def is_macos():
    return sys.platform == "darwin"


def dataloader_num_workers(requested: int) -> int:
    """Cap DataLoader workers on macOS (spawn + MPS is sensitive to high worker counts)."""
    if requested < 0:
        return 0
    if is_macos():
        return min(requested, 4)
    return requested


def _mps_available():
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def get_preferred_device():
    """Apple Silicon: MPS if available, else CPU. This codebase targets macOS only."""
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device(use_cuda=True, device_id=None, usage=5):
    """
    Return torch.device. use_cuda=False forces CPU.
    When True (default), use Apple MPS if available, else CPU.
    device_id and usage are ignored (kept for call-site compatibility).
    """
    if not use_cuda:
        return torch.device("cpu")
    return get_preferred_device()


def set_device(usage=5):
    """Print and rely on MPS or CPU for training (usage is unused; kept for script compatibility)."""
    if _mps_available():
        print("Using device: mps (Apple GPU)")
    else:
        print("Using device: cpu")


def default_device(use_cuda=True):
    """If use_cuda, prefer MPS then CPU; otherwise CPU."""
    if not use_cuda:
        return torch.device("cpu")
    return get_preferred_device()


def to_device(b, device=None, non_blocking=False):
    """
    Recursively put `b` on `device`
    components of b are torch tensors
    """
    if device is None:
        device = default_device(use_cuda=True)

    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_device(o, device) for o in b)

    out = b.to(device)
    # MPS: avoid BatchNorm/autograd layout bugs (e.g. channels_last / rank mismatch on backward)
    if device.type == "mps":
        out = out.contiguous()
    return out


def to_numpy(b):
    """
    Components of b are torch tensors
    """
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    return b.detach().cpu().numpy()


def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + list(add)


class GetAttr:

    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra

    def _dir(self): 
        return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None: return getattr(attr,k)
        # raise AttributeError(k)

    def __dir__(self): 
        return custom_dir(self,self._dir())

#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): 
        self.__dict__.update(data)

