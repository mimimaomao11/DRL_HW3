import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer

episodes = 1000
batch_size = 32
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.1
lr = 1e-3

env = GridWorld(mode="player")
model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=lr)
buffer = ReplayBuffer()

reward_history = []
loss_history = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(30):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = model(state_tensor).argmax().item()

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) > batch_size:
            s, a, r, s_, d = buffer.sample(batch_size)

            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s_ = torch.FloatTensor(s_)
            d = torch.FloatTensor(d)

            q_values = model(s)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            # 🔥 Double DQN 核心
            with torch.no_grad():
                next_actions = model(s_).argmax(1)   # online 選
                max_q_next = target_model(s_).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)                        # target 評估

                target = r + gamma * max_q_next * (1 - d)

            loss = nn.MSELoss()(q_value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        if done:
            break

    reward_history.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 50 == 0:
        target_model.load_state_dict(model.state_dict())

    if ep % 100 == 0:
        print(f"[Double] Episode {ep}, Reward: {total_reward:.2f}")

# 存圖
plt.plot(reward_history)
plt.savefig("results/plots/reward_double.png")
plt.clf()

plt.plot(loss_history)
plt.savefig("results/plots/loss_double.png")

print("Double DQN training done")

np.savetxt("results/logs/double_rewards.txt", reward_history)

torch.save(model.state_dict(), "results/models/double.pth")