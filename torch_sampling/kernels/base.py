from typing import Protocol, Any, Tuple
import torch

class Kernel(Protocol):
    def init_state(self, target, x: torch.Tensor) -> Any:
        ...

    def step(self, target, state: Any, step_size: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        ...