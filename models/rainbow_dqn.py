import torch
import torch.nn as nn
import torch.nn.functional as F
from models.noisy_layer import NoisyLinear

class RainbowDQN(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(RainbowDQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        
        # Value stream with NoisyNets
        self.value_hidden = NoisyLinear(128, 128)
        self.value_out = NoisyLinear(128, num_atoms)
        
        # Advantage stream with NoisyNets
        self.adv_hidden = NoisyLinear(128, 128)
        self.adv_out = NoisyLinear(128, out_dim * num_atoms)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        feat = self.feature(x)
        
        v = self.value_hidden(feat)
        v = F.relu(v)
        v = self.value_out(v)
        v = v.view(batch_size, 1, self.num_atoms)
        
        a = self.adv_hidden(feat)
        a = F.relu(a)
        a = self.adv_out(a)
        a = a.view(batch_size, self.out_dim, self.num_atoms)
        
        # Dueling integration
        q = v + a - a.mean(dim=1, keepdim=True)
        
        # Return probabilities using softmax (Categorical DQN)
        prob = F.softmax(q, dim=-1)
        return prob
        
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
