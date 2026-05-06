import numpy as np
import random

class PERBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_p = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_p)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (state, action, reward, next_state, done)
            self.priorities[idx] = max_p

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