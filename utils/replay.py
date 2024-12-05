import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state), 
                torch.tensor(action), 
                torch.tensor(reward), 
                torch.cat(next_state),
                torch.tensor(done))
    
    def __len__(self):
        return len(self.buffer)