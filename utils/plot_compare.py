import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(path):
    if path.endswith(".npy"):
        return np.load(path)
    else:
        return np.loadtxt(path)

def smooth(y, window=20):
    return np.convolve(y, np.ones(window)/window, mode='valid')

data = {
    "DQN": "results/logs/dqn_rewards.txt",
    "Double": "results/logs/double_rewards.txt",
    "Dueling": "results/logs/dueling_rewards.txt",
    "Keras": "results/logs/keras_rewards.npy",
    "Rainbow": "results/logs/rainbow_rewards.txt"
}

plt.figure(figsize=(10,6))

for name, path in data.items():
    try:
        rewards = load_data(path)
        rewards = smooth(rewards)   # 🔥 關鍵
        plt.plot(rewards, label=name)
    except:
        print(f"Missing {name}")

plt.title("DQN Variants Comparison (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid()

plt.savefig("results/plots/all_models_smooth.png")
plt.show()