class HyperparameterScheduler:
    def __init__(self):
        self.initial_lr = 1e-4
        self.min_lr = 1e-6
        self.lr_decay_factor = 0.5
        self.lr_patience = 200
        
        self.initial_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
    
    def update(self, avg_reward, agent):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        if self.episodes_without_improvement >= self.lr_patience:
            current_lr = agent.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.lr_decay_factor, self.min_lr)
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.episodes_without_improvement = 0
            return True
        return False