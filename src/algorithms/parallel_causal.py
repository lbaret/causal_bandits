import numpy as np

from src.models import Parallel

from .base_algorithm import BaseAlgorithm


class ParallelCausal(BaseAlgorithm):
    label = "Algorithm 1"

    def __init__(self) -> None:
        super().__init__()
    
    def run(self, T, model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        h = T / 2
        for t in range(int(h)):
            x, y = model.sample(model.K-1) # do nothing
            xij = np.hstack((1-x, x, 1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y * xij
            
        self.infrequent = self.estimate_infrequent(h)
        n = int(float(h)/len(self.infrequent))
        self.trials[self.infrequent] = n # note could be improved by adding to rather than reseting observation results - does not change worst case. 
        self.success[self.infrequent] = model.sample_multiple(self.infrequent, n)
        self.u = np.true_divide(self.success, self.trials)
        self.r = self.u - model.get_costs()
        self.best_action = self.argmax_rand(self.r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
   
            
    def estimate_infrequent(self, h):
        qij_hat = np.true_divide(self.trials, h)
        s_indx = np.argsort(qij_hat) #indexes of elements from s in sorted(s)
        m_hat = Parallel.calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent