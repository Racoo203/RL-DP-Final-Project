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

def objective(trial: Trial, env, algorithm_func, n_seeds=5):
    # Suggest parameters
    params = {
        "alpha": trial.suggest_float("alpha", 0.0001, 0.5, log = True),
        "gamma": trial.suggest_float("gamma", 0.8, 0.9999),
        "epsilon": 1.0,
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9, 0.9999),
        "epsilon_min": 0.01,
        "n_episodes": 100
    }
    
    seed_scores = []
    
    for seed in range(n_seeds):
        opt_log.info(f"Trial {trial.number} | Seed {seed} | Starting | Params: {params}")

        _, rewards = algorithm_func(env, params = params, show_progress = False, seed = seed)
        
        seed_auc = np.sum(rewards)
        seed_scores.append(seed_auc)
        
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

        opt_log.info(f"Trial {trial.number} | Seed {seed} | Ended | Score: {params}")
        
    mu = np.mean(seed_scores)
    sigma = np.std(seed_scores)
    
    opt_log.info(f"Trial {trial.number} | Scores: ({mu}, {sigma})")

    return mu, sigma

def param_opt_pipeline(algorithm, env, n_trials = 100):

    # pruner = optuna.pruners.MedianPruner(n_startup_trials = 5, n_warmup_steps = 1)

    study = optuna.create_study(
        storage = "sqlite:///param_opt.sqlite3",
        study_name = algorithm.__name__,
        load_if_exists = True,
        directions=["maximize", "minimize"],
    )

    with tqdm(total = n_trials, desc = f"Optimizing {algorithm.__name__}") as pbar:
        
        def callback(study, trial):
            pbar.update(1)
            # pbar.set_postfix({"Best Reward": f"{study.best_value:.2f}"})

        study.optimize(
            # Pass show_progress=False to the algorithm here
            lambda trial: objective(trial, env, algorithm), 
            n_trials = n_trials,
            n_jobs = -1,
            callbacks = [callback]
        )

    return study.best_trials