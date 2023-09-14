import numpy as np

from .base_algorithm import BaseAlgorithm


class UCB(BaseAlgorithm):
    """ 
    Implements Generic UCB algorithm.
    """
    def __init__(self) -> None:
        super().__init__()

    def run(self, T, model):
        if T <= model.K: # result is not defined if the horizon is shorter than the number of actions
            self.best_action = None
            return np.nan
        
        actions = range(0,model.K)
        self.trials = np.ones(model.K)
        self.success = model.sample_multiple(actions, 1)
        
        for t in range(model.K,T):
            arm = self.argmax_rand(self.upper_bound(t))
            self.trials[arm] += 1
            self.success[arm] += model.sample_multiple(arm, 1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = self.argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]