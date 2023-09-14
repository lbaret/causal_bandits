import numpy as np
from .base_algorithm import BaseAlgorithm


class ObservationalEstimate(BaseAlgorithm):
    """ Just observes for all actions, and then selects the arm with the best emprical mean. 
        Assumes P(Y|do(X)) = P(Y|X) as for ParallelCausal. Some actions may be entirely unexplored. """
    label = "Observational"
    
    def __init__(self) -> None:
        super().__init__()
    
    def run(self, T, model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)

        for t in range(T):
            x, y = model.sample(model.K-1)
            xij = np.hstack((1-x, x, 1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y * xij
        
        self.u = np.true_divide(self.success, self.trials)
        self.best_action = self.argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]