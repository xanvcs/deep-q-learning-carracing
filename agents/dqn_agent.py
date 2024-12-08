import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from models.dqn import DQN
from utils.replay import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_shape, n_actions, scheduler):
        self.device = device
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.scheduler = scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=scheduler.initial_lr)
        self.memory = ReplayBuffer(100000)
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = scheduler.initial_epsilon
        self.epsilon_min = scheduler.min_epsilon
        self.epsilon_decay = scheduler.epsilon_decay
        self.target_update = 10

    @torch.inference_mode()
    def select_action(self, state, eval_mode=False):
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                action_idx = self.policy_net(state).max(1)[1].item()
        else:
            action_idx = random.randrange(5)
            
        # Convert discrete actions to continuous
        if action_idx == 0:    # Do nothing
            return np.array([0.0, 0.0, 0.0])
        elif action_idx == 1:  # Steer left
            return np.array([-1.0, 0.0, 0.0])
        elif action_idx == 2:  # Steer right
            return np.array([1.0, 0.0, 0.0])
        elif action_idx == 3:  # Gas
            return np.array([0.0, 1.0, 0.0])
        else:                  # Brake
            return np.array([0.0, 0.0, 0.8])
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        expected_q_values = reward + (1 - done.float()) * self.gamma * next_q_values
        
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)