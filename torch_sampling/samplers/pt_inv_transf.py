import math
from typing import NamedTuple

import torch
from tqdm import tqdm

from ..kernels import HMCKernel, MALAKernel, MHKernel
from ..targets.conditional_target import ConditionalTarget
from ..targets.gaussian import Gaussian
from ..targets.target_distribution import TargetDistribution
from ..utils.kernel_state import KernelState
from .base import Sampler
from .pt import PT

MIN_TIME = 1e-10
TOL = 1e5


class PTInvTransfState(NamedTuple):
    inner_state: KernelState
    z: torch.Tensor
    time: torch.Tensor


class PTInverseTransform(Sampler):
    def __init__(
        self,
        target: TargetDistribution,
        time_start: float = 1e-4,
        jump_steps: int = 0,
        jump_beta_schedule: torch.Tensor = None,
        jump_ref_std: float = 1.0,
        jump_step_mode: str = "mala",
        jump_step_size: float = 0.1,
        jump_swap_mode: str = "nrpt",
        jump_swap_every: int = 1,
        jump_leapfrog_steps: int = 5,
        jump_adaptation_rate: float = 0.05,
        jump_target_acceptance: float = None,
        compile: bool = True,
        verbose: bool = False,
    ):
        super().__init__(target=target, verbose=verbose)

        self.time_start = float(time_start)
        if self.time_start < 0.0:
            raise ValueError("time_start must be >= 0.")

        if jump_steps > 0 and jump_beta_schedule is None:
            raise ValueError("jump_beta_schedule must be provided when jump_steps > 0.")

        self.jump_steps = jump_steps
        self.jump_ref_std = jump_ref_std
        self.jump_beta_schedule = jump_beta_schedule
        self.jump_step_mode = jump_step_mode
        self.jump_step_size = jump_step_size
        self.jump_swap_mode = jump_swap_mode
        self.jump_swap_every = jump_swap_every
        self.jump_leapfrog_steps = jump_leapfrog_steps
        self.jump_adaptation_rate = jump_adaptation_rate
        self.jump_target_acceptance = jump_target_acceptance

        self._compile = compile

        self._current_state: PTInvTransfState = None

    def _inverse_transform_step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        
        x0 = (x - (1.0 - time) * z) / (time + 1e-10)
        return x0

    def _build_state(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> PTInvTransfState:
        time_clamp = torch.clamp(time, min=MIN_TIME, max=1.0)
        inv_time_clamp = 1.0 / time_clamp
        x0_next = z + inv_time_clamp * (x - z)
        grad_log_prob_x, log_prob_x = self.target.grad_log_prob(x0_next, return_log_prob=True)
        grad_log_prob_x = torch.clamp(grad_log_prob_x, min=-TOL, max=TOL)

        inner = KernelState(
            x=x,
            log_prob=log_prob_x,
            grad=grad_log_prob_x,
        )
        return PTInvTransfState(
            inner_state=inner,
            z=z,
            time=time,
        )

    def _init_state(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        time: torch.Tensor,
    ) -> None:
        self._current_state = self._build_state(x=x, z=z, time=time)

    def _perform_jump(
        self,
        time: torch.Tensor,
        return_log_acceptance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        jump_step_size = self.jump_step_size if self.jump_step_size is not None else 0.1

        x = self._current_state.inner_state.x
        z = self._current_state.z

        n_replicas = len(self.jump_beta_schedule)
        z_replicas = z.unsqueeze(1).repeat(1, n_replicas, 1)
        z_replicas_flat = z_replicas.reshape(-1, self.target.dim)

        cond_target = ConditionalTarget(self.target, time, z_replicas_flat).to(x.device)
        mean = z.mean(dim=0)
        std = self.jump_ref_std * torch.ones(self.target.dim, device=x.device) / math.sqrt(time.item() + 1e-3)
        ref_target = Gaussian(
            mean=mean,
            std=std,
        )

        if self.jump_step_mode == "mala":
            jump_kernel = MALAKernel(noise_scale=time.item())
        elif self.jump_step_mode == "hmc":
            jump_kernel = HMCKernel(n_leapfrog_steps=self.jump_leapfrog_steps, momentum_scale=time.item())
        else:
            jump_kernel = MHKernel()

        pt_sampler = PT(
            target=cond_target,
            reference=ref_target,
            kernel=jump_kernel,
            beta_schedule=self.jump_beta_schedule,
            step_size=jump_step_size,
            swap_mode=self.jump_swap_mode,
            swap_every=self.jump_swap_every,
            adaptation_rate=self.jump_adaptation_rate,
            target_acceptance=self.jump_target_acceptance,
            verbose=self.verbose,
            compile=self._compile,
        ).to(x.device)

        x0 = z.clone()
        out = pt_sampler(x0, n_steps=self.jump_steps, return_log_acceptance=return_log_acceptance)

        if return_log_acceptance:
            samples, log_acceptance = out
        else:
            samples = out
            log_acceptance = None

        new_x = samples[:, -1, :]

        self._init_state(
            x=new_x,
            z=z,
            time=time,
        )

        return new_x, log_acceptance

    def forward(
        self,
        x0: torch.Tensor,
        z: torch.Tensor = None,
        return_trajectory: bool = False,
        return_jump_log_acceptance: bool = False,
    ) -> torch.Tensor:
        time_start = torch.tensor(self.time_start, device=x0.device, dtype=x0.dtype)

        if z is None:
            z = x0.clone()
        x = x0.clone()

        if return_trajectory:
            xs = [x.clone().to("cpu")]

        self._init_state(
            x=x,
            z=z,
            time=time_start,
        )

        if self.jump_steps > 0:
            x, jump_log_acceptance = self._perform_jump(
                time_start,
                return_log_acceptance=return_jump_log_acceptance,
            )
            if return_trajectory:
                xs.append(x.clone().to("cpu"))

        if self.verbose:
            pbar = tqdm(total=1, desc="PT Inverse Transform")

        current = self._current_state
        x = self._inverse_transform_step(
            x=current.inner_state.x,
            z=current.z,
            time=current.time
        )
        self._current_state = self._build_state(x=x, z=current.z, time=torch.tensor(1.0, device=x0.device, dtype=x0.dtype))

        if return_trajectory:
            xs.append(x.clone().to("cpu"))
        if self.verbose:
            pbar.update(1)

        if self.verbose:
            pbar.close()

        res = []
        if return_trajectory:
            res.append(torch.stack(xs, dim=0))
        else:
            res.append(x)

        if return_jump_log_acceptance:
            if self.jump_steps > 0:
                res.append(jump_log_acceptance)
            else:
                res.append(None)

        return res[0] if len(res) == 1 else tuple(res)

    def sample_trajectory(
        self,
        n_samples: int = 1,
        z: torch.Tensor = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32,
        )
        return self.forward(x0=x0, z=z, return_trajectory=True)

    def sample(
        self,
        n_samples: int = 1,
        z: torch.Tensor = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        x0 = self.build_initial_point(
            n_samples=n_samples,
            device=device,
            dtype=torch.float32,
        )
        return self.forward(x0=x0, z=z, return_trajectory=False)
