from omegaconf import DictConfig, OmegaConf

from torch_sampling.targets import (
    TargetDistribution,
    LennardJones,
    ALDPVacuum,
    ALDP_OMM,
    MLPPosterior
)

from torch_sampling.utils.targets import (
    uniform_gaussian_mixture,
    nonuniform_gaussian_mixture
)

def build_target(config: DictConfig) -> TargetDistribution:
    """Build the target object based on the config."""

    params_dict = OmegaConf.to_container(config.task.params, resolve=True, throw_on_missing=True)
    
    if config.task.name in ["gm_2_40", "gm_16_40", "gm_32_80", "gm_160_165"]:
        target = uniform_gaussian_mixture(
            **params_dict
        )
    elif config.task.name in ["gmnu_2_40", "gmnu_16_40"]:
        target = nonuniform_gaussian_mixture(
            **params_dict
        )
    elif "lennard_jones" in config.task.name:
        target = LennardJones(
            **params_dict            
        )
    elif config.task.name == "aldp_vacuum":
        target = ALDPVacuum(
            **params_dict
        )
    elif config.task.name == "aldp_implicit_omm":
        target = ALDP_OMM(
            **params_dict
        )
    elif config.task.name == "mlp_posterior":
        target = MLPPosterior(
            **params_dict
        )
    else:
        raise ValueError(f"Unknown task: {config.task.name}")
    
    return target