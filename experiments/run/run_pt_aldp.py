# nohup python -u run_pt_aldp.py > run_pt_aldp.log 2>&1 &

import os
import itertools
import numpy as np
from run_utils import run_jobs, adjust_budget, prepare_grid_for_product, normalize_combo_tags


GPUS = [0, 1]

LOGGER = "wandb"
NUM_FOLDS = 3

OUTPUT_DIR = "/home/fran/work_fran/sampling/experiments/run/output"


grid_list = [
    {
        "tags": ["sota_comparison"],
        "logger.project": ["sampling_arxiv"],
        "task": ["aldp_vacuum"],
        "task.n_samples_gen": [100000],
        "task.params.regularize_energy": [True],
        "save_samples": [True],
        "n_chunks": [1],
        "sampler": ["parallel_tempering"],
        "sampler.save_diagnostics": [True],
        "budget": np.logspace(4, 6, num=8, dtype=int).tolist(),
        "sampler.params.beta_schedule.type": ["geometric"],
        "sampler.params.beta_schedule.min_val": [0.3],
        "sampler.params.beta_schedule.n_replicas": [5],
        "sampler.params.beta_schedule.optimize": [False],
        "sampler.params.kernel.name": ["hmc"],
        "sampler.params.kernel.n_leapfrog_steps": [5],
        "sampler.params.step_size": [0.000001],
        "sampler.params.swap_mode": ["nrpt"],
        "sampler.params.swap_every": [1],
        "sampler.params.adaptation_rate": [0.05],
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