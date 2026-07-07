import wandb

from omegaconf import DictConfig, ListConfig, OmegaConf


def _normalize_tags(raw_tags) -> list[str]:
    """Normalize tags into a flat list of non-empty strings."""

    if raw_tags is None:
        return []

    if isinstance(raw_tags, str):
        return [raw_tags] if raw_tags.strip() else []

    if isinstance(raw_tags, (list, tuple, ListConfig)):
        normalized = []
        for tag in raw_tags:
            if tag is None:
                continue
            tag_str = str(tag).strip()
            if tag_str:
                normalized.append(tag_str)
        return normalized

    tag_str = str(raw_tags).strip()
    return [tag_str] if tag_str else []


def retrieve_tags(config: DictConfig) -> list[str]:
    """Retrieve tags from config."""

    tags = []
    for key in config:
        if isinstance(config[key], DictConfig):
            tags.extend(retrieve_tags(config[key]))
        elif key == 'tags':
            tags.extend(_normalize_tags(config[key]))
        else:
            pass

    return tags

def init_logger(config: DictConfig) -> None:
    """
    Initialize the logger for the experiment.
    
    Args:
        config (DictConfig): Configuration for the experiment.
    """

    if config.logger.name not in ["wandb", "none"]:
        raise ValueError(f"Unsupported logger: {config.logger.name}. Supported loggers: ['wandb', 'none']")

    if config.logger.name == "none":
        print("No logger initialized. Skipping logging.")
        return
    
    # Initialize Weights & Biases
    wandb.init(
        project=config.logger.project,
        tags=retrieve_tags(config),
    )
    
    # Log the configuration
    wandb.config.update(OmegaConf.to_container(config, resolve=True))
    
    print("Wandb logger initialized with configuration:")
    print(OmegaConf.to_yaml(config))