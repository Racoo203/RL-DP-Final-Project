import numpy as np

import optuna
from optuna.trial import Trial
import gymnasium as gym

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

def get_params(trial: Trial, alg_name):
    """
    Returns only the hyperparameters relevant to the specific algorithm.
    """
    # Common parameters
    params = {
        # Problem parameters: <- Fixed parameters to the given problem
        "n_episodes": 300,
        "gamma": 0.99,

        # Config parameters: <- Display and Robustness testing
        "show_progress": False,
    }

    # Algorithm parameters: <- Parameters optimized for robust algorithms
    if alg_name in ["alg_SARSA", "alg_Q", "alg_nStep_SARSA", "alg_SARSA_Lambda"]:
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 0.5, log = True)
        params["epsilon"] = 1.0
        # params["floor_fraction"] = trial.suggest_float("floor_fraction", 0.4, 0.8)
        params["epsilon_min"] = 0.01

        # decay_steps = int(params["floor_fraction"] * params["n_episodes"])
        # params["epsilon_decay"] = params["epsilon_min"] ** (1.0 / decay_steps)
        params["epsilon_decay"] = trial.suggest_float("alpha", 0.9, 0.999, log = True)


    if alg_name == "alg_nStep_SARSA":
        params["n"] = trial.suggest_int("n", 2, 14)

    if alg_name == "alg_SARSA_Lambda":
        params["lambda"] = trial.suggest_float("lambda", 0.0, 1.0)
    
    if alg_name == "alg_REINFORCE_B":
        params["alpha_theta"] = trial.suggest_float("alpha_theta", 1e-4, 1e-2, log=True)
        params["alpha_w"] = trial.suggest_float("alpha_w", 1e-4, 0.3, log=True)
        params["n_episodes"] = 600
        
    return params

def objective(trial: Trial, algorithm_func, n_seeds = 7):
    local_env = gym.make("Acrobot-v1")
    params = get_params(trial, algorithm_func.__name__)
    seed_scores = []
    
    for seed in range(n_seeds):
        params["seed"] = seed
        opt_log.info(f"Trial {trial.number} | Seed {seed} | Starting | Params: {params}")

        _, rewards_history = algorithm_func(local_env, params = params)
        
        last_n = int(len(rewards_history) * 0.1)
        score = np.mean(rewards_history[-last_n:])
        seed_scores.append(score)

        opt_log.info(f"Trial {trial.number} | Seed {seed} | Ended | Score: {score}")
        
        trial.report(np.mean(seed_scores), step = seed)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    iqm = get_iqm(seed_scores)
    std = np.std(seed_scores)

    opt_log.info(f"Trial {trial.number} | Score: {iqm} | Std: {std}")

    return iqm

def param_opt_pipeline(algorithm, n_trials=64):

    storage = optuna.storages.RDBStorage(
        url="sqlite:///param_opt.sqlite3",
        engine_kwargs={"connect_args": {"timeout": 30}}
    )

    study = optuna.create_study(
        storage=storage,
        study_name=algorithm.__name__,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    # --- Clean up incomplete trials ---
    finished_states = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    ]

    incomplete = [t for t in study.trials if t.state not in finished_states]

    if incomplete:
        opt_log.info(f"{algorithm.__name__} | Removing {len(incomplete)} incomplete trials: "
                     f"{[t.number for t in incomplete]}")
        
        storage_obj = optuna.storages.get_storage(storage)
        for t in incomplete:
            storage_obj.set_trial_state_values(
                t._trial_id,
                state=optuna.trial.TrialState.FAIL
            )
        opt_log.info(f"{algorithm.__name__} | Cleanup complete")

    finished = [t for t in study.trials if t.state in finished_states]
    n_finished = len(finished)
    n_remaining = max(0, n_trials - n_finished)

    opt_log.info(f"{algorithm.__name__} | Finished: {n_finished} | Target: {n_trials} | Remaining: {n_remaining}")

    if n_remaining == 0:
        opt_log.info(f"{algorithm.__name__} | Target reached, returning best trial immediately")
        
        return study.best_trial
    else:
        with tqdm(total=n_remaining, desc=f"Optimizing {algorithm.__name__}",
                initial=0) as pbar:

            def callback(study, trial):
                pbar.update(1)

            study.optimize(
                lambda trial: objective(trial, algorithm),
                n_trials=n_remaining,
                callbacks=[callback],
                n_jobs = 6
            )

        return study.best_trial