import pickle
from collections import Counter, defaultdict


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.logger import LogManager

sns.set_theme(style="darkgrid")

manager = LogManager()
main_log = manager.get_logger("Main", "main.log")

# --- MODEL SAVING ---

def save_policy(q_table, name):
    """Saves the Q-table using pickle."""
    path = f"models/{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(dict(q_table), f, pickle.HIGHEST_PROTOCOL)

    main_log.info(f"Model checkpoint saved: {path}")

def load_policy(name):
    with open(f"models/{name}.pkl", "rb") as f:
        data = pickle.load(f)
    
    sample = next(iter(data.values()))
    n = len(sample)
    
    if isinstance(sample, np.ndarray) and sample.sum() < 1.1:  # probability array = REINFORCE
        default = defaultdict(lambda: np.full(n, 1.0 / n))
    else:  # Q-values
        default = defaultdict(lambda: np.zeros(n))
    
    default.update(data)
    return default

# --- RESULTS PLOTTING --- 

def plot_learning_curve(rewards, label):
    """Standardized plotting for learning curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Learning Curve: {label}")
    plt.legend()
    plt.show()

    main_log.info(f"Plot created in notebook.")

def moving_avg(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_smoothed_learning_curve(rewards, name):
    plt.figure(figsize=(10,5))
    plt.plot(moving_avg(rewards), label = "Smoothed rewards")
    plt.title(f"Smoothed Learning Curve: {name}")
    plt.legend()
    plt.show()

def plot_state_visits(visited_states):
    counts = Counter(visited_states)
    values = list(counts.values())
    plt.hist(values, bins=50)

def compute_entropy(policy):
    entropies = []
    for probs in policy.values():
        entropies.append(-np.sum(probs * np.log(probs + 1e-8)))
    return np.mean(entropies)

def q_stats(Q):
    values = np.concatenate(list(Q.values()))
    return np.mean(values), np.std(values)