import numpy as np
import random
from collections import deque

class NStepPERBuffer:
    def __init__(self, capacity=10000, alpha=0.6, n_step=3, gamma=0.95):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_step)

    def get_n_step(self):
        # Calculate n-step return
        res_reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            res_reward += (self.gamma ** i) * transition[2] # reward
            
        # Extract state, action from first transition
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        
        # Extract next_state, done from last transition
        next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
        return state, action, res_reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        n_state, n_action, n_reward, n_next_state, n_done = self.get_n_step()
        
        max_p = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((n_state, n_action, n_reward, n_next_state, n_done))
            self.priorities.append(max_p)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (n_state, n_action, n_reward, n_next_state, n_done)
            self.priorities[idx] = max_p
            
        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + 1e-5

    def __len__(self):
        return len(self.buffer)
