import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import hydra
import wandb
import numpy as np

from omegaconf import DictConfig, OmegaConf

from utils import (
    build_target,
    build_sampler,
    run_sampler,
    evaluate_samples,
    plot_results,
    init_logger,
) 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

def fix_config(config: DictConfig) -> None:
    """Fix config to have the correct types and values."""
    
    for key, value in config.items():
        if isinstance(value, str):
            if value.lower() == 'none':
                config[key] = None
            elif value.lower() == 'true':
                config[key] = True
            elif value.lower() == 'false':
                config[key] = False
            elif value.isdigit():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except ValueError:
                    pass
    
    return config

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config : DictConfig) -> None:

    print(f"PyTorch version: {torch.__version__}", )

    print(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    config = fix_config(config)

    print('Config:')
    print(OmegaConf.to_yaml(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize logger
    init_logger(config)

    # Build target distribution
    target = build_target(config)
    target = target.to(device)
    print(f"Target distribution:\n{target}")

    # Build sampler
    sampler = build_sampler(config, target, device)
    sampler = sampler.to(device)
    print(f"Sampler:\n{sampler}")

    print("Running sampler...")
    pred_samples = run_sampler(config, sampler, device)
    pred_samples = pred_samples.to(device)

    target = target.to("cpu")
    gt_samples = target.sample(config.task.n_samples_gt)
    gt_samples = gt_samples.to(device)

    if config.evaluate_samples:
        print("Evaluating samples...")
        metrics = evaluate_samples(gt_samples, pred_samples, target, device)
        metrics = {f"test/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        print('Metrics:')
        for key, value in metrics.items():
            print(f"{key}: {value}")

        if wandb.run is not None:
            wandb.log(metrics)
    
    should_plot = config.task.plot if hasattr(config.task, 'plot') else False
    if should_plot:
        print('Plotting results...')
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        plot_results(
            target, 
            gt_samples.detach(),
            pred_samples.detach(),
            output_dir=output_dir,
            device=device,
            config=config
        )
    
    if config.save_samples:
        print('Saving samples...')
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Could not create output directory: {e}")
        output_path = f"{output_dir}/pred_samples.npy"
        pred_samples_np = pred_samples.cpu().numpy()
        np.save(output_path, pred_samples_np)
        if wandb.run is not None:
            wandb.save(output_path, base_path=output_dir)
        print(f"Saved predicted samples to {output_path}")

    print('Done!')

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()