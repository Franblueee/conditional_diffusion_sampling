# nohup python -u run_mala_mlp_posterior.py > run_mala_mlp_posterior.log 2>&1 &

import os
import itertools
import numpy as np
from run_utils import run_jobs, adjust_budget, prepare_grid_for_product, normalize_combo_tags

GPUS = [3]

LOGGER = "wandb"
NUM_FOLDS = 3

OUTPUT_DIR = "/home/fran/work_fran/sampling/experiments/run/output"


grid_list = [
    {
        "tags": ["sota_comparison"],
        "logger.project": ["sampling_arxiv"],
        "task": ["mlp_posterior"],
        "sampler": ["mala"],
        "budget": np.logspace(3, 5, num=10, dtype=int).tolist(),
        "sampler.params.step_size": [0.00001],
        "sampler.params.adaptation_rate": [0.01],
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
            job_list.append(combo)

run_jobs(
    job_list,
    GPUS,
    LOGGER,
    OUTPUT_DIR
)