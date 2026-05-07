import torch
import torch.optim as optim
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld
from models.rainbow_dqn import RainbowDQN
from utils.n_step_buffer import NStepPERBuffer

print("=" * 50)
print("🌈 Full Rainbow DQN Training (Double + Dueling + PER + Noisy + C51 + N-Step)")
print("=" * 50)

env = GridWorld(mode="random")

num_atoms = 51
v_min = -10.0
v_max = 10.0
support = torch.linspace(v_min, v_max, num_atoms)

model = RainbowDQN(num_atoms=num_atoms, v_min=v_min, v_max=v_max)
target_model = RainbowDQN(num_atoms=num_atoms, v_min=v_min, v_max=v_max)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_step = 3
gamma = 0.95
buffer = NStepPERBuffer(n_step=n_step, gamma=gamma)

episodes = 1000
batch_size = 32

rewards = []

def projection_distribution(next_state, rewards, dones):
    batch_size = next_state.size(0)
    
    with torch.no_grad():
        target_model.reset_noise()
        model.reset_noise()
        
        # Double DQN: online network selects action
        next_dist = model(next_state)
        next_q_expected = (next_dist * support).sum(dim=2)
        next_action = next_q_expected.argmax(1)
        
        # Target network evaluates
        next_target_dist = target_model(next_state)
        next_target_dist = next_target_dist[range(batch_size), next_action]
        
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (gamma ** n_step) * support.unsqueeze(0)
        Tz = Tz.clamp(v_min, v_max)
        b = (Tz - v_min) / ((v_max - v_min) / (num_atoms - 1))
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix when l == u
        l[(u > 0) & (l == u)] -= 1
        u[(l < (num_atoms - 1)) & (l == u)] += 1
        
        proj_dist = torch.zeros(next_target_dist.size())
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
        
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_target_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_target_dist * (b - l.float())).view(-1))
        
        return proj_dist

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(40):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Noisy Nets replace epsilon-greedy
        model.reset_noise()
        with torch.no_grad():
            dist = model(state_tensor)
            q_expected = (dist * support).sum(dim=2)
            action = q_expected.argmax(1).item()

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) > batch_size:
            s, a, r, s_, d, w, idx = buffer.sample(batch_size)

            s = torch.FloatTensor(np.array(s))
            a = torch.LongTensor(np.array(a))
            r = torch.FloatTensor(np.array(r))
            s_ = torch.FloatTensor(np.array(s_))
            d = torch.FloatTensor(np.array(d))
            w = torch.FloatTensor(np.array(w))

            proj_dist = projection_distribution(s_, r, d)

            model.reset_noise()
            dist = model(s)
            dist = dist[range(batch_size), a]
            
            # Cross entropy loss
            loss = -(proj_dist * torch.log(dist + 1e-5)).sum(dim=1)
            
            # PER update
            buffer.update_priorities(idx, loss.detach().numpy())
            
            # Total loss with IS weights
            loss = (w * loss).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if done:
            break

    rewards.append(total_reward)
    
    if ep % 50 == 0:
        target_model.load_state_dict(model.state_dict())

    if ep % 100 == 0:
        print(f"[Full Rainbow] Episode {ep}, Reward: {total_reward:.2f}")

os.makedirs("results/logs", exist_ok=True)
np.savetxt("results/logs/rainbow_rewards.txt", rewards)

print("Full Rainbow training done")
torch.save(model.state_dict(), "results/models/rainbow.pth")