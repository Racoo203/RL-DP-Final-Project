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

def objective(trial: Trial, env, algorithm_func, n_seeds = 10):
    # Suggest parameters
    params = {
        "alpha": trial.suggest_float("alpha", 0.001, 0.3, log = True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "epsilon": 1.0,
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9, 0.999),
        "epsilon_min": 0.05,
        "n_episodes": 500
    }
    
    seed_scores = []
    
    for seed in range(n_seeds):
        opt_log.info(f"Trial {trial.number} | Seed {seed} | Starting | Params: {params}")

        _, rewards = algorithm_func(env, params = params, show_progress = False, seed = seed)
        
        score = np.sum(rewards)
        seed_scores.append(score)

        opt_log.info(f"Trial {trial.number} | Seed {seed} | Ended | Score: {score}")
        
        # Report intermediate value for pruning
        trial.report(np.mean(seed_scores), step = seed)
        
        # Allow pruner to stop unpromising trials
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    iqm = get_iqm(seed_scores)
    std = np.std(seed_scores)

    opt_log.info(f"Trial {trial.number} | Score: {iqm} | Std: {std}")

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