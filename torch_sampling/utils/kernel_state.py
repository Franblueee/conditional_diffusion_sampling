import torch
from typing import NamedTuple, Optional

class KernelState(NamedTuple):
    x: torch.Tensor
    log_prob: torch.Tensor
    grad: Optional[torch.Tensor] = None