import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()

        # shared feature
        self.feature = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU()
        )

        # value stream
        self.value = nn.Sequential(
            nn.Linear(128, 1)
        )

        # advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.feature(x)

        value = self.value(x)
        advantage = self.advantage(x)

        # 🔥 核心：去掉平均值（穩定）
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q