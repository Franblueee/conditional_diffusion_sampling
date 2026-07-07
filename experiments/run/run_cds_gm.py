# nohup python -u run_pisde_gm.py > run_pisde_gm.log 2>&1 &

import os
import itertools
import numpy as np
from run_utils import run_jobs, adjust_budget, prepare_grid_for_product, normalize_combo_tags

GPUS = [3, 4, 6]

LOGGER = "wandb"
NUM_FOLDS = 3

OUTPUT_DIR = "/home/fran/work_fran/sampling/experiments/run/output"

grid_list = [
    {
        "task": ["gm_2_40", "gmnu_2_40"],
        "tags": ["sota_comparison"],
        "logger.project": ["sampling_arxiv"],
        "sampler": ["cds"],
        "budget": np.logspace(3, 5, num=10, dtype=int).tolist(),
        "integration_budget": [10, 100, 1000],
        "sampler.params.time_schedule.type": ["geometric"],
        "sampler.params.time_schedule.min_val": [0.001, 0.01],
        "sampler.params.noise_schedule.base_noise_var": [0.5],
        "sampler.params.noise_schedule.type": ["constant"],
        "sampler.params.corrector_mode": ["mala"],
        "sampler.params.corrector_steps": [0],
        "sampler.params.corrector_adaptation_rate": [0.0],
        "sampler.params.corrector_step_size": [0.1],
        "sampler.params.jump_ref_std": [1.0],
        "sampler.params.jump_step_mode": ["mala"],
        "sampler.params.jump_leapfrog_steps": [0],
        "sampler.params.jump_beta_schedule.type": ["geometric"],
        "sampler.params.jump_beta_schedule.min_val": [0.001],
        "sampler.params.jump_beta_schedule.n_replicas": [5],
        "sampler.params.jump_adaptation_rate": [0.05],
    },
    {
        "task": ["gm_16_40", "gmnu_16_40"],
        "tags": ["sota_comparison"],
        "logger.project": ["sampling_arxiv"],
        "sampler": ["cds"],
        "budget": np.logspace(3, 6, num=10, dtype=int).tolist(),
        "integration_budget": [10, 100, 1000],
        "sampler.params.time_schedule.type": ["geometric"],
        "sampler.params.time_schedule.min_val": [0.001, 0.01],
        "sampler.params.noise_schedule.base_noise_var": [0.5],
        "sampler.params.noise_schedule.type": ["constant"],
        "sampler.params.corrector_mode": ["mala"],
        "sampler.params.corrector_steps": [0],
        "sampler.params.corrector_adaptation_rate": [0.0],
        "sampler.params.corrector_step_size": [0.1],
        "sampler.params.jump_ref_std": [1.0],
        "sampler.params.jump_step_mode": ["mala"],
        "sampler.params.jump_leapfrog_steps": [0],
        "sampler.params.jump_beta_schedule.type": ["geometric"],
        "sampler.params.jump_beta_schedule.min_val": [0.001],
        "sampler.params.jump_beta_schedule.n_replicas": [10],
        "sampler.params.jump_adaptation_rate": [0.05],
    }
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".log"):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not remove file {file_path}: {e}")

job_list = []
for grid in grid_list:
    for _ in range(NUM_FOLDS):
        grid_for_product = prepare_grid_for_product(grid)

        keys, values = zip(*grid_for_product.items())
        for v in itertools.product(*values):
            combo = dict(zip(keys, v))
            combo = normalize_combo_tags(combo)
            combo = adjust_budget(combo)
            if combo is not None:
                job_list.append(combo)

run_jobs(
    job_list,
    GPUS,
    LOGGER,
    OUTPUT_DIR
)