import subprocess
from queue import Queue
from threading import Thread
import time
import os
import sys


def prepare_grid_for_product(grid):
    prepared_grid = {}
    for key, value in grid.items():
        if key == "tags" and isinstance(value, list) and all(isinstance(tag, str) for tag in value):
            prepared_grid[key] = [value]
        else:
            prepared_grid[key] = value
    return prepared_grid


def normalize_combo_tags(combo):
    if "tags" not in combo:
        return combo

    tag_value = combo["tags"]
    if tag_value is None:
        combo["tags"] = []
    elif isinstance(tag_value, str):
        combo["tags"] = [tag_value] if tag_value.strip() else []
    elif isinstance(tag_value, (list, tuple)):
        combo["tags"] = [str(tag).strip() for tag in tag_value if tag is not None and str(tag).strip()]
    else:
        tag_as_str = str(tag_value).strip()
        combo["tags"] = [tag_as_str] if tag_as_str else []

    return combo

def adjust_budget(combo):
    sampler_name = combo["sampler"]

    if sampler_name == "cds":

        corrector_steps = combo["sampler.params.corrector_steps"]
        corrector_adaptation_rate = combo["sampler.params.corrector_adaptation_rate"]
        if corrector_steps == 0 and corrector_adaptation_rate > 0:
            print(f"Invalid config: corrector_adaptation_rate={corrector_adaptation_rate} > 0 but corrector_steps={corrector_steps} == 0. Skipping this config.")
            return None

        budget = combo.pop("budget")
        integration_budget = combo.pop("integration_budget")
        n_replicas = combo["sampler.params.jump_beta_schedule.n_replicas"]

        integration_steps = integration_budget // (1 + corrector_steps)

        jump_budget = max(budget - integration_budget, 0)

        jump_kernel_name = combo["sampler.params.jump_step_mode"]
        jump_leapfrog_steps = combo["sampler.params.jump_leapfrog_steps"]

        if kernel_name != "hmc" and n_leapfrog_steps > 0:
            print(f"Invalid config: jump_leapfrog_steps={jump_leapfrog_steps} > 0 but jump_step_mode={jump_kernel_name} is not 'hmc'. Skipping this config.")
            return None
        
        if jump_kernel_name == "hmc" and jump_leapfrog_steps == 0:
            print(f"Invalid config: jump_leapfrog_steps={jump_leapfrog_steps} but jump_step_mode={jump_kernel_name} is 'hmc'. Skipping this config.")
            return None

        if jump_kernel_name != "hmc":
            jump_steps = jump_budget // n_replicas
        else:
            jump_steps = jump_budget // (n_replicas * jump_leapfrog_steps)

        combo["sampler.params.jump_steps"] = jump_steps
        combo["sampler.integration_steps"] = integration_steps
        
    elif sampler_name == "parallel_tempering":
        
        n_leapfrog_steps = combo["sampler.params.kernel.n_leapfrog_steps"]
        kernel_name = combo["sampler.params.kernel.name"]

        if kernel_name != "hmc" and n_leapfrog_steps > 0:
            print(f"Invalid config: n_leapfrog_steps={n_leapfrog_steps} > 0 but kernel.name={kernel_name} is not 'hmc'. Skipping this config.")
            return None
    
        if kernel_name == "hmc" and n_leapfrog_steps == 0:
            print(f"Invalid config: n_leapfrog_steps={n_leapfrog_steps} but kernel.name={kernel_name} is 'hmc'. Skipping this config.")
            return None

        budget = combo.pop("budget")
        n_replicas = combo["sampler.params.beta_schedule.n_replicas"]
        if kernel_name != "hmc":
            n_steps_pt = budget // n_replicas
        else:
            n_steps_pt = budget // (n_replicas * n_leapfrog_steps)
        combo["sampler.n_steps"] = n_steps_pt
    elif sampler_name == "smc":

        n_leapfrog_steps = combo["sampler.params.kernel.n_leapfrog_steps"]
        kernel_name = combo["sampler.params.kernel.name"]

        if kernel_name != "hmc" and n_leapfrog_steps > 0:
            print(f"Invalid config: n_leapfrog_steps={n_leapfrog_steps} > 0 but kernel.name={kernel_name} is not 'hmc'. Skipping this config.")
            return None
    
        if kernel_name == "hmc" and n_leapfrog_steps == 0:
            print(f"Invalid config: n_leapfrog_steps={n_leapfrog_steps} but kernel.name={kernel_name} is 'hmc'. Skipping this config.")
            return None

        budget = combo.pop("budget")
        n_particles = combo["sampler.params.n_particles"]
        if kernel_name != "hmc":
            n_steps_smc = budget // n_particles
        else:
            n_steps_smc = budget // (n_particles * n_leapfrog_steps)
        n_steps_smc = max(n_steps_smc, 2)
        combo["sampler.n_steps"] = n_steps_smc
    elif sampler_name == "hmc":
        budget = combo.pop("budget")
        n_leapfrog_steps = combo["sampler.params.n_leapfrog_steps"]
        n_steps_hmc = budget // n_leapfrog_steps
        combo["sampler.n_steps"] = n_steps_hmc
    elif sampler_name == "mala":
        budget = combo.pop("budget")
        combo["sampler.n_steps"] = budget
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    return combo


def run_jobs(
    job_list, 
    gpus, 
    logger, 
    output_dir
):
    
    job_queue = Queue()
    for job in job_list:
        job_queue.put(job)

    print(f"Total jobs to run: {job_queue.qsize()}")

    def worker(gpu_id):

        while not job_queue.empty():
            
            try:
                config_dict = job_queue.get_nowait()
            except Queue.Empty:
                break

            try:                
                
                overrides = [f"{k}={v}" for k, v in config_dict.items()]
                cmd = [sys.executable, "-u", "/home/fran/work_fran/sampling/experiments/main.py"] \
                    + overrides \
                    + [f"logger={logger}"]
                
                task_name = config_dict["task"]
                sampler_name = config_dict["sampler"]
                timestamp = int(time.time())
                log_filename = f"{output_dir}/{sampler_name}_{task_name}_gpu{gpu_id}_{timestamp}.log"

                current_env = os.environ.copy()
                current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                current_env["CUDA_LAUNCH_BLOCKING"] = "1"
                current_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

                print(f"[GPU {gpu_id}] Remaining jobs: {job_queue.qsize()}")
                print(f"[GPU {gpu_id}] Starting job with config:\n{config_dict}")
                
                with open(log_filename, "w") as log_file:
                    subprocess.run(
                        cmd, 
                        env=current_env,
                        check=False,
                        stdout=log_file,
                        stderr=subprocess.STDOUT
                    )
            except Exception as e:
                print(f"[GPU {gpu_id}] Error with config {config_dict}: {e}")
            finally:
                job_queue.task_done()
    
    threads = []
    for gpu in gpus:
        t = Thread(target=worker, args=(gpu,))
        t.start()
        threads.append(t)
        time.sleep(5)