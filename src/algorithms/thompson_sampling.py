import numpy as np

from .base_algorithm import BaseAlgorithm


class ThompsonSampling(BaseAlgorithm):
    """ Sample actions via the Thomson sampling approach and return the empirically best arm 
        when the number of rounds is exhausted """
    label = "Thompson Sampling"

    def __init__(self) -> None:
        super().__init__()
    
    def run(self, T, model):
        self.trials = np.full(model.K, 2, dtype=int)
        self.success = np.full(model.K, 1, dtype=int)
        
        for t in range(T):
            fails = self.trials - self.success
            theta = np.random.beta(self.success, fails)
            arm = self.argmax_rand(theta)
            self.trials[arm] +=1
            self.success[arm] += model.sample_multiple(arm, 1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = self.argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]