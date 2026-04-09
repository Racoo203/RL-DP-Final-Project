import pickle

import matplotlib.pyplot as plt
import seaborn as sns

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
    """Loads a Q-table from a pickle file."""
    with open(f"models/{name}.pkl", "rb") as f:
        return pickle.load(f)

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