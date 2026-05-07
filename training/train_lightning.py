import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, IterableDataset

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gridworld import GridWorld
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer

print("=" * 60)
print("DQN Training with PyTorch Lightning")
print("=" * 60)

# 超參數
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.98
EPSILON_MIN = 0.1
LEARNING_RATE = 1e-3
UPDATE_FREQ = 50  # 每多少steps更新target network

class DQNLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for DQN training"""
    
    def __init__(self, lr=LEARNING_RATE, gamma=GAMMA, batch_size=BATCH_SIZE):
        super().__init__()
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.buffer = ReplayBuffer(capacity=10000)
        self.epsilon = EPSILON_START
        self.training_step_count = 0
        
        self.reward_history = []
        self.loss_history = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        s, a, r, s_, d = batch
        
        # 計算Q值
        q_values = self.model(s)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        
        # 計算target Q值
        with torch.no_grad():
            max_q_next = self.target_model(s_).max(1)[0]
            target = r + self.gamma * max_q_next * (1 - d)
        
        # MSE loss
        loss = nn.MSELoss()(q_value, target)
        
        self.log('train_loss', loss, prog_bar=True)
        self.loss_history.append(loss.item())
        
        # 每UPDATE_FREQ步更新target network
        self.training_step_count += 1
        if self.training_step_count % UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def train_dataloader(self):
        return DQNDataLoader(self.buffer, self.model, self.batch_size, 
                            num_episodes=EPISODES, epsilon_decay=EPSILON_DECAY)


class DQNDataLoader(IterableDataset):
    """自定義IterableDataset用於DQN訓練"""
    
    def __init__(self, buffer, model, batch_size, num_episodes=1000, epsilon_decay=0.98):
        self.buffer = buffer
        self.model = model
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.epsilon_decay = epsilon_decay
        
        self.env = GridWorld(mode="random")
        self.epsilon = EPSILON_START
        self.episode_count = 0
        self.reward_history = []
        
    def __iter__(self):
        for ep in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            for t in range(30):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        action = self.model(state_tensor).argmax().item()
                
                next_state, reward, done = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                # 返回batch進行訓練
                if len(self.buffer) > self.batch_size:
                    s, a, r, s_, d = self.buffer.sample(self.batch_size)
                    
                    s = torch.FloatTensor(s)
                    a = torch.LongTensor(a)
                    r = torch.FloatTensor(r)
                    s_ = torch.FloatTensor(s_)
                    d = torch.FloatTensor(d)
                    
                    yield (s, a, r, s_, d)
                
                if done:
                    break
            
            self.reward_history.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, EPSILON_MIN)
            
            if ep % 100 == 0:
                avg_reward = np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(self.reward_history)
                print(f"[Lightning] Episode {ep}/{self.num_episodes}, Reward: {total_reward:.2f}, "
                      f"Avg(100): {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")


# 訓練
if __name__ == "__main__":
    # 創建模型
    lightning_module = DQNLightningModule(
        lr=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE
    )
    
    # 設置callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='results/models/',
        filename='dqn_lightning-{epoch:02d}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )
    
    # 創建Trainer
    trainer = Trainer(
        max_epochs=1,  # 1個epoch = 1個完整訓練循環
        devices=1,
        accelerator='cpu' if not torch.cuda.is_available() else 'gpu',
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=True
    )
    
    # 開始訓練
    print("\nStarting PyTorch Lightning training...\n")
    
    # 用自定義DataLoader進行訓練
    dataloader = DQNDataLoader(
        lightning_module.buffer,
        lightning_module.model,
        BATCH_SIZE,
        num_episodes=EPISODES,
        epsilon_decay=EPSILON_DECAY
    )
    
    # 手動訓練循環（因為我們使用custom IterableDataset）
    optimizer = optim.Adam(lightning_module.model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    
    loss_history = []
    reward_history = []
    
    for ep in range(EPISODES):
        state = dataloader.env.reset()
        total_reward = 0
        epsilon = max(EPSILON_START * (EPSILON_DECAY ** ep), EPSILON_MIN)
        
        for t in range(30):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = lightning_module.model(state_tensor).argmax().item()
            
            next_state, reward, done = dataloader.env.step(action)
            lightning_module.buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if len(lightning_module.buffer) > BATCH_SIZE:
                s, a, r, s_, d = lightning_module.buffer.sample(BATCH_SIZE)
                
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                s_ = torch.FloatTensor(s_)
                d = torch.FloatTensor(d)
                
                q_values = lightning_module.model(s)
                q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    max_q_next = lightning_module.target_model(s_).max(1)[0]
                    target = r + GAMMA * max_q_next * (1 - d)
                
                loss = nn.MSELoss()(q_value, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lightning_module.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_history.append(loss.item())
            
            if done:
                break
        
        reward_history.append(total_reward)
        scheduler.step()
        
        if ep % 50 == 0:
            target_model_state = dict(lightning_module.target_model.named_parameters())
            model_state = dict(lightning_module.model.named_parameters())
            lightning_module.target_model.load_state_dict(lightning_module.model.state_dict())
        
        if ep % 100 == 0:
            avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            print(f"Episode {ep}/{EPISODES}, Reward: {total_reward:.2f}, "
                  f"Avg(100): {avg_reward:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 保存模型
    print("\n💾 Saving Lightning model...")
    torch.save(lightning_module.model.state_dict(), "results/models/dqn_lightning.pth")
    
    # 繪製和保存圖表
    print("📊 Saving plots...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards (PyTorch Lightning)')
    plt.grid(True)
    plt.savefig("results/plots/reward_lightning.png", dpi=100)
    
    plt.subplot(1, 2, 2)
    if loss_history:
        plt.plot(loss_history)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss (PyTorch Lightning)')
        plt.grid(True)
    plt.savefig("results/plots/loss_lightning.png", dpi=100)
    
    plt.close()
    
    # 保存rewards到npy
    np.save("results/logs/lightning_rewards.npy", reward_history)
    
    print("\n" + "=" * 60)
    print("✅ PyTorch Lightning DQN training completed!")
    print(f"✅ Total episodes: {len(reward_history)}")
    print(f"✅ Final reward: {reward_history[-1]:.2f}")
    print(f"✅ Average reward (last 100): {np.mean(reward_history[-100:]):.2f}")
    print(f"✅ Max reward: {max(reward_history):.2f}")
    print(f"✅ Models saved to: results/models/")
    print(f"✅ Plots saved to: results/plots/")
    print("=" * 60)