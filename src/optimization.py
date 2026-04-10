import numpy as np

import optuna
from optuna.trial import Trial

import threading
from tqdm import tqdm

from src.logger import LogManager

# --- GLOBAL CONSTANTS ---
OPTIMIZATION_DB_PATH = "sqlite:///param_opt.sqlite3"
OPTIMIZATION_LOGS_PATH = "logs/optimization.log"

manager = LogManager()
opt_log = manager.get_logger(name = "Optuna", file_name = "optimization.log", to_console = False)

optuna.logging.set_verbosity(optuna.logging.ERROR)

def get_iqm(sample):
    sample = np.array(sample)
    q1 = np.quantile(sample, 0.25)
    q3 = np.quantile(sample, 0.75)
    return np.mean(sample[(sample >= q1) & (sample <= q3)])

def objective(trial: Trial, env, algorithm_func, n_seeds=5):
    # Suggest parameters
    params = {
        "alpha": trial.suggest_float("alpha", 0.0001, 0.5, log = True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.9999),
        "epsilon": 1.0,
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9, 0.9999),
        "epsilon_min": 0.01,
        "n_episodes": 400
    }
    
    seed_scores = []
    
    for seed in range(n_seeds):
        opt_log.info(f"Trial {trial.number} | Seed {seed} | Starting | Params: {params}")

        _, rewards = algorithm_func(env, params = params, show_progress = False, seed = seed)
        
        seed_auc = np.sum(rewards)
        seed_scores.append(seed_auc)

        opt_log.info(f"Trial {trial.number} | Seed {seed} | Ended | Score: {params}")
        
    iqm = get_iqm(seed_scores)
    opt_log.info(f"Trial {trial.number} | Score: {iqm}")

    return iqm

def param_opt_pipeline(algorithm, env, n_trials = 64):

    study = optuna.create_study(
        storage = "sqlite:///param_opt.sqlite3",
        study_name = algorithm.__name__,
        load_if_exists = True,
        direction = "maximize",
        pruner = optuna.pruners.MedianPruner()
    )

    with tqdm(total = n_trials, desc = f"Optimizing {algorithm.__name__}") as pbar:
        
        def callback(study, trial):
            pbar.update(1)

        study.optimize(
            lambda trial: objective(trial, env, algorithm), 
            n_trials = n_trials,
            # n_jobs = -1,
            callbacks = [callback]
        )

        # n_jobs helps with doing multiple trials at once,
        # but sacrifices reproducibility. Therefore, while slower,
        # one job at a time will be done.

    return study.best_trial