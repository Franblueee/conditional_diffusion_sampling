import torch
import numpy as np
import hydra

from torch_sampling.targets import TargetDistribution, Gaussian, InfUniform

from torch_sampling.kernels import (
    Kernel,
    MHKernel,
    MALAKernel,
    HMCKernel
)

from torch_sampling.samplers import (
    Sampler,
    LA,
    MALA,
    HMC,
    PT,
    SMC,
    CDS
)
from torch_sampling.utils import fix_chirality
from torch_sampling.samplers.smc import compute_log_g

from .plot import plot_smc_diagnostics, plot_pt_diagnostics

from omegaconf import DictConfig, OmegaConf

def sampler_forward_wrapper(
    sampler: Sampler,
    x0: torch.torch.Tensor,
    config: DictConfig,
    device: torch.device = torch.device("cpu")
):
    """Wrapper function to call the sampler's sample method."""

    if isinstance(sampler, CDS):

        z = x0.clone()

        return_jump_log_acceptance = config.sampler.save_diagnostics
        if not return_jump_log_acceptance:
            print(f"Running {sampler.__class__.__name__} sampler without jump diagnostics...")
            samples = sampler(x0=x0, z=z, return_jump_log_acceptance=False)
        else:
            print(f"Running {sampler.__class__.__name__} sampler with jump diagnostics...")
            samples, log_acceptances = sampler(x0=x0, z=z, return_jump_log_acceptance=True)
            if log_acceptances is not None:
                acc_probs = torch.exp(torch.clamp(log_acceptances, max=0.0)) # (n_replicas-1,)
                rejection_rates = 1.0 - acc_probs  # (n_replicas-1,)
                rejection_rates_extended = torch.cat([torch.tensor([0.0], device=rejection_rates.device), rejection_rates], dim=0)  # add 0.0 for the first replica
                Lambda = torch.cumsum(rejection_rates_extended, dim=0) # (n_replicas,)
                plot_pt_diagnostics(
                    schedules_list=[sampler.jump_beta_schedule],
                    Lambda_list=[Lambda],
                    rejection_rates_list=[rejection_rates],
                    output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    name="jump_diagnostics"
                )
        return samples
    elif isinstance(sampler, PT):
        
        n_steps = config.sampler.n_steps
        # Optimize beta schedule if specified in the config
        if config.sampler.params.beta_schedule.optimize:
            
            n_steps_tune = n_steps // 2
            n_steps = n_steps - n_steps_tune

            print("Running chains to determine the optimal beta schedule...")
            x0_opt = x0.clone()

            n_rounds = int(np.ceil(np.log2(n_steps_tune)))-1

            schedules_list, Lambda_list, rejection_rates_list = sampler.run_rounds(
                x0=x0_opt,
                n_samples=x0_opt.shape[0],
                n_rounds=n_rounds,
                device=device
            )

            plot_pt_diagnostics(
                schedules_list=schedules_list,
                Lambda_list=Lambda_list,
                rejection_rates_list=rejection_rates_list,
                output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                name="pt_schedule_optimization_diagnostics"
            )

            print("Optimal beta schedule determined.")

        return_log_acceptance = config.sampler.save_diagnostics
        output = sampler(x0, n_steps=n_steps, return_log_acceptance=return_log_acceptance)
        if not return_log_acceptance:
            samples = output # (n_samples, n_replicas, dim)
            samples = samples[:, -1, :]  # Take samples from the coldest chain
        if return_log_acceptance:
            samples, log_acceptances = output # (n_samples, n_replicas, dim), (n_steps, n_replicas-1)
            samples = samples[:, -1, :]  # Take samples from the coldest chain
            acc_probs = torch.exp(torch.clamp(log_acceptances, max=0.0)) # (n_replicas-1,)
            rejection_rates = 1.0 - acc_probs  # (n_replicas-1,)
            rejection_rates_extended = torch.cat([torch.tensor([0.0], device=rejection_rates.device), rejection_rates], dim=0)  # add 0.0 for the first replica
            Lambda = torch.cumsum(rejection_rates_extended, dim=0) # (n_replicas,)
            plot_pt_diagnostics(
                schedules_list=[sampler.beta_schedule],
                Lambda_list=[Lambda],
                rejection_rates_list=[rejection_rates],
                output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                name="pt_final_diagnostics"
            )           
        return samples
    elif isinstance(sampler, SMC):

        # Optimize beta schedule if specified in the config
        if config.sampler.params.beta_schedule.optimize:

            n_steps_tune = sampler.beta_schedule.shape[0] // 2

            # Run rounds to determine the optimal beta schedule
            print("Running rounds to determine the optimal beta schedule...")
            x0_opt = x0.clone()

            n_rounds = int(np.ceil(np.log2(n_steps_tune)))-1

            schedules_list, Lambda_list, sqrt_D_list = sampler.run_rounds(
                x0=x0_opt,
                n_samples=x0_opt.shape[0], 
                n_rounds=n_rounds,
                max_schedule_length=n_steps_tune,
                final_schedule_length=n_steps_tune,
                device=device
            )

            plot_smc_diagnostics(
                schedules_list=schedules_list,
                Lambda_list=Lambda_list,
                sqrt_D_list=sqrt_D_list,
                output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                name="smc_schedule_optimization_diagnostics"
            )

            print("Optimal beta schedule determined.")


        if config.sampler.save_diagnostics:
            return_log_weights = True
            return_log_increments = True
        else:
            return_log_weights = False
            return_log_increments = False
        
        output = sampler(x0, return_log_weights=return_log_weights, return_log_increments=return_log_increments)

        if config.sampler.save_diagnostics:
            samples, log_weights, log_increments = output # (n_samples, dim), (n_steps-1,), (n_steps-1,)
        
            log_g_1 = compute_log_g(log_weights, log_increments, exponent=1) # (n_steps-1,)
            log_g_2 = compute_log_g(log_weights, log_increments, exponent=2) # (n_steps-1,)

            D = log_g_2 - 2*log_g_1 # (n_steps-1,)
            D = torch.clamp(D, min=0.0) # (n_steps-1,)
            sqrt_D = torch.sqrt(D) # (n_steps-1,)
            sqrt_D = torch.cat([torch.tensor([0.0], device=sqrt_D.device, dtype=sqrt_D.dtype), sqrt_D], dim=0) # (n_steps,)

            Lambda = torch.cumsum(sqrt_D, dim=0) # (n_steps,)
            plot_smc_diagnostics(
                schedules_list=[sampler.beta_schedule],
                Lambda_list=[Lambda],
                sqrt_D_list=[sqrt_D],
                output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                name="smc_final_diagnostics"
            )
        else:
            samples = output
        return samples
    elif isinstance(sampler, (LA, MALA, HMC)):
        n_steps = config.sampler.n_steps
        return sampler(x0, n_steps=n_steps)
    else:
        raise ValueError(f"Sampler type {type(sampler)} not supported in the wrapper.")


def run_sampler(config: DictConfig, sampler: Sampler, device: torch.device) -> torch.Tensor:
    """Run the sampler to generate samples."""
    
    n_samples = config.task.n_samples_gen

    # Ensure the sampler is on the correct device
    sampler = sampler.to(device)

    # Load initial samples if provided, otherwise generate random initial samples
    if config.task.init_samples_path == "random":
        print(f"Generating {n_samples} random initial samples...")
        dim = sampler.target.dim
        x0 = torch.randn(n_samples, dim).to(device)
    else:
        try:
            print(f"Loading initial samples from {config.task.init_samples_path}...")
            x0 = torch.load(config.task.init_samples_path).to(device)
            if x0.shape[0] > n_samples:
                x0 = x0[:n_samples]
            elif x0.shape[0] < n_samples:
                # repeat samples to reach n_samples
                n_repeats = n_samples // x0.shape[0] + 1
                x0 = x0.repeat(n_repeats, 1)[:n_samples]
            x0 = x0.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load initial samples from {config.task.init_samples_path}: {e}")
    
    # Generate samples using the sampler forward wrapper
    # Use chunking if n_samples is large to avoid out-of-memory issues
    samples_list = []
    n_chunks = config.n_chunks if config.n_chunks is not None else 1
    samples_per_chunk = n_samples // n_chunks
    for i in range(n_chunks):
        idx = np.random.choice(x0.shape[0], samples_per_chunk, replace=False)
        x0_chunk = x0[idx]
        chunk_samples = sampler_forward_wrapper(
            sampler=sampler,
            x0=x0_chunk,
            config=config,
            device=device
        )
        samples_list.append(chunk_samples)
    samples = torch.cat(samples_list, dim=0)
    print(f"Generated {samples.shape[0]} samples in {n_chunks} chunks.")

    # For molecular targets, fix chirality of samples if specified in the config
    if config.task.name in ["aldp_vacuum", "aldp_implicit_omm"]:
        print("Fixing chirality of samples...")
        samples = samples.view(samples.shape[0], 22, 3)
        samples, n_flipped = fix_chirality(samples, target_sign='positive')
        samples = samples.view(samples.shape[0], -1)
        print(f"Number of samples with flipped chirality corrected: {n_flipped}/{samples.shape[0]}")        

    return samples

def build_interpolation_schedule(
    type: str, 
    n_steps: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    p: float = 1.0
):
    if type == "linear":
        schedule = torch.linspace(
            min_val,
            max_val,
            n_steps
        )
    elif type == "polynomial":
        schedule = torch.linspace(
            min_val,
            max_val,
            n_steps
        )
        schedule = min_val + (max_val - min_val) * (schedule ** p)
    elif type == "geometric":
        r = (max_val / min_val) ** (1.0 / (n_steps - 1))
        schedule = min_val * (r ** torch.arange(n_steps))
    else:
        raise ValueError(f"Unknown schedule type: {type}")

    schedule = torch.clamp(schedule, min=min_val, max=max_val)
    
    return schedule

def build_noise_schedule(
    base_noise_var: float,
    time_schedule: torch.Tensor,
    type: str = "linear",
    symmetric: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Compute a noise schedule tensor for a discretized interval (0, 1).

    Arguments:
        base_noise_var: base noise variance
        type: type of noise schedule ("constant", "linear", "quadratic", "cosine")
        symmetric: whether to use symmetric noise schedule
        device: the torch device to place the schedule on

    Returns:
        noise_vars: A tensor of shape (n_steps,) containing the noise variances.
    """

    if type == "constant":
        noise_vars = torch.full_like(time_schedule, base_noise_var)
    
    elif type == "linear":
        if symmetric:
            noise_vars = base_noise_var * time_schedule * (1 - time_schedule) * 4
        else:
            noise_vars = base_noise_var * time_schedule
            
    elif type == "quadratic":
        time_sq = time_schedule ** 2
        if symmetric:
            noise_vars = base_noise_var * time_sq * (1 - time_sq) * 4
        else:
            noise_vars = base_noise_var * time_sq
            
    elif type == "cosine":
        f_cos = 0.5 * (1.0 - torch.cos(np.pi * time_schedule))
        if symmetric:
            noise_vars = base_noise_var * f_cos * (1 - f_cos) * 4
        else:
            noise_vars = base_noise_var * f_cos
            
    else:
        raise ValueError(f"Invalid type: {type}. Supported types are 'constant', 'linear', 'quadratic', and 'cosine'.")
    
    return noise_vars

def build_kernel(config: DictConfig) -> Kernel:
    """Build the kernel based on the config."""

    name = config.name

    if name == "mh":
        kernel = MHKernel()
    elif name == "mala":
        kernel = MALAKernel()
    elif name == "hmc":
        n_leapfrof_steps = config.get('n_leapfrog_steps', 5)
        kernel = HMCKernel(n_leapfrog_steps=n_leapfrof_steps)
    else:
        raise ValueError(f"Unknown kernel: {config.name}")
    
    return kernel

def build_sampler(config: DictConfig, target: TargetDistribution, device: torch.device = None) -> Sampler:
    """Build the sampler based on the config."""
    
    if device is None:
        device = torch.device("cpu")

    params_dict = OmegaConf.to_container(config.sampler.params, resolve=True, throw_on_missing=True)

    if config.sampler.name == "la":
        sampler = LA(
            target=target,
            **params_dict,
        )
    elif config.sampler.name == "mala":
        sampler = MALA(
            target=target,
            **params_dict,
        )
    elif config.sampler.name == "hmc":
        sampler = HMC(
            target=target,
            **params_dict
        )
    elif config.sampler.name == "parallel_tempering":

        reference_type = config.sampler.params.reference.type
        if reference_type == "gaussian":
            dim = target.dim
            mean = torch.zeros(dim, device=device)
            std = torch.ones(dim, device=device) * config.sampler.params.reference.std
            reference = Gaussian(mean=mean, std=std)
        elif reference_type == "inf_uniform":
            dim = target.dim
            reference = InfUniform(dim=dim)
        else:
            raise ValueError(f"Unknown reference type: {reference_type}")

        beta_schedule = build_interpolation_schedule(
            type=config.sampler.params.beta_schedule.get('type', 'linear'),
            n_steps=config.sampler.params.beta_schedule.get('n_replicas', 10),
            min_val=config.sampler.params.beta_schedule.get('min_val', 0.01),
            max_val=config.sampler.params.beta_schedule.get('max_val', 1.0),
            p=config.sampler.params.beta_schedule.get('p', 1.0)
        )

        # remove the first if it is 0.0
        if beta_schedule[0] == 0.0:
            beta_schedule = beta_schedule[1:]
        
        print(f"Using beta schedule: {beta_schedule}")

        kernel = build_kernel(config.sampler.params.kernel)

        # remove beta_schedule from params_dict
        params_dict.pop('reference', None)
        params_dict.pop('beta_schedule', None)
        params_dict.pop('kernel', None)

        sampler = PT(
            target=target,
            reference=reference,
            beta_schedule=beta_schedule,
            kernel=kernel,
            **params_dict
        )
    elif config.sampler.name == "smc":

        beta_schedule = build_interpolation_schedule(
            type=config.sampler.params.beta_schedule.get('type', 'linear'),
            n_steps=config.sampler.n_steps,
            min_val=config.sampler.params.beta_schedule.get('min_val', 0.0),
            max_val=config.sampler.params.beta_schedule.get('max_val', 1.0),
            p=config.sampler.params.beta_schedule.get('p', 1.0)
        )

        kernel = build_kernel(config.sampler.params.kernel)

        params_dict.pop('beta_schedule', None)
        params_dict.pop('kernel', None)

        sampler = SMC(
            target=target,
            beta_schedule=beta_schedule,
            kernel=kernel,
            **params_dict
        )
    elif config.sampler.name == "cds":

        jump_steps = config.sampler.params.jump_steps
        integration_steps = config.sampler.integration_steps

        if jump_steps == 0:
            beta_schedule = None
        else:
            n_replicas = config.sampler.params.jump_beta_schedule.n_replicas
            if n_replicas == 1:
                beta_schedule = torch.tensor([config.sampler.params.jump_beta_schedule.max_val])
            else:
                beta_schedule = build_interpolation_schedule(
                        type=config.sampler.params.jump_beta_schedule.type,
                        n_steps=n_replicas,
                        min_val=config.sampler.params.jump_beta_schedule.min_val,
                        max_val=config.sampler.params.jump_beta_schedule.max_val,
                        p=config.sampler.params.jump_beta_schedule.p
                    )
        
        if integration_steps < 2:
            time_schedule = torch.tensor([config.sampler.params.time_schedule['min_val']])
        else:
            time_schedule = build_interpolation_schedule(
                type=config.sampler.params.time_schedule['type'],
                n_steps=integration_steps,
                min_val=config.sampler.params.time_schedule['min_val'],
                max_val=config.sampler.params.time_schedule['max_val'],
                p=config.sampler.params.time_schedule.get('p', 1.0)
            )

        noise_schedule = build_noise_schedule(
            base_noise_var=config.sampler.params.noise_schedule.base_noise_var,
            time_schedule=time_schedule,
            type=config.sampler.params.noise_schedule.type,
            symmetric=config.sampler.params.noise_schedule.symmetric
        )

        params_dict.pop('jump_beta_schedule', None)
        params_dict.pop('noise_schedule', None)
        params_dict.pop('time_schedule', None)
        params_dict.pop('jump_steps', None)

        sampler = CDS(
            target=target,
            time_schedule=time_schedule,
            noise_schedule=noise_schedule,
            jump_steps=jump_steps,
            jump_beta_schedule=beta_schedule,
            **params_dict
        )
    else:
        raise ValueError(f"Unknown sampler: {config.sampler.name}")
    
    return sampler
