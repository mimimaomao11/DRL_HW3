import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld
from models.dueling_dqn import DuelingDQN
from utils.per_buffer import PERBuffer

env = GridWorld(mode="random")

model = DuelingDQN()
target_model = DuelingDQN()
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
buffer = PERBuffer()

gamma = 0.95
epsilon = 1.0

episodes = 1000

rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(40):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = model(state_tensor).argmax().item()

        next_state, reward, done = env.step(action)

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) > 32:
            s, a, r, s_, d, w, idx = buffer.sample(32)

            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s_ = torch.FloatTensor(s_)
            d = torch.FloatTensor(d)
            w = torch.FloatTensor(w)

            # Double DQN
            next_actions = model(s_).argmax(1)
            next_q = target_model(s_).gather(1, next_actions.unsqueeze(1)).squeeze()

            target = r + gamma * next_q * (1 - d)

            q = model(s).gather(1, a.unsqueeze(1)).squeeze()

            loss = (w * (q - target.detach())**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            errors = (q - target).detach().numpy()
            buffer.update_priorities(idx, errors)

        if done:
            break

    rewards.append(total_reward)
    epsilon = max(epsilon * 0.995, 0.1)

    if ep % 100 == 0:
        print(f"[Rainbow] Episode {ep}, Reward: {total_reward}")

np.savetxt("results/logs/rainbow_rewards.txt", rewards)

print("Rainbow training done")