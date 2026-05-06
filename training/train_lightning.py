import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer

print("=" * 50)
print("DQN Training (PyTorch Standard)")
print("=" * 50)

# 超參數
episodes = 1000
batch_size = 32
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.1
lr = 1e-3

# 初始化
env = GridWorld(mode="random")
model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
buffer = ReplayBuffer()

reward_history = []
loss_history = []

print("\nStarting training...")

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

            with torch.no_grad():
                max_q_next = target_model(s_).max(1)[0]
                target = r + gamma * max_q_next * (1 - d)

            loss = nn.MSELoss()(q_value, target)

            optimizer.zero_grad()
            loss.backward()
            
            # Bonus: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            loss_history.append(loss.item())

        if done:
            break

    reward_history.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if ep % 50 == 0:
        target_model.load_state_dict(model.state_dict())

    # Bonus: Learning Rate Scheduling
    scheduler.step()

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# 保存模型
print("\nSaving model...")
torch.save(model.state_dict(), "results/models/dqn_final.pth")

# 繪製和保存圖表
print("Saving plots...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards')
plt.grid(True)
plt.savefig("results/plots/reward_dqn_final.png", dpi=100)

plt.subplot(1, 2, 2)
if loss_history:
    plt.plot(loss_history)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
plt.savefig("results/plots/loss_dqn_final.png", dpi=100)

plt.close()

print("\n" + "=" * 50)
print("✓ DQN training completed!")
print(f"✓ Total episodes: {len(reward_history)}")
print(f"✓ Final reward: {reward_history[-1]:.2f}")
print(f"✓ Average reward (last 100): {np.mean(reward_history[-100:]):.2f}")
print("=" * 50)