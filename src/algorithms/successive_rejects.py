import numpy as np

from .base_algorithm import BaseAlgorithm


class SuccessiveRejects(BaseAlgorithm):
    """ Implementation based on the paper 'Best Arm Identification in Multi-Armed Bandits',Audibert,Bubeck & Munos"""
    label = "Successive Reject"

    def __init__(self) -> None:
        super().__init__()
    
    def run(self, T, model):
        if T <= model.K:
            self.best_action = None
            return np.nan
        else:
            self.trials = np.zeros(model.K)
            self.success = np.zeros(model.K)
            self.actions = range(0, model.K)
            self.allocations = self.allocate(T, model.K)
            self.rejected = np.zeros((model.K), dtype=bool)
            
            for k in range(0,model.K-1):
                nk = self.allocations[k]
                self.success[self.actions] += model.sample_multiple(self.actions,nk)
                self.trials[self.actions] += nk
                self.reject()
            
            assert len(self.actions) == 1, f"number of arms remaining is: {self.actions}, not 1."
            assert sum(self.trials) <= T, f"number of pulls = {self.trials}, exceeds T = {T}"
            
            self.best_action = self.actions[0]
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
    
    def allocate(self, T, K):
        logK = .5 + np.true_divide(1, range(2, K+1)).sum()
        n = np.zeros((K), dtype=int)
        n[1:] =  np.ceil((1.0 / logK) * np.true_divide((T - K), range(K, 1, -1)))
        allocations = np.diff(n)
        return allocations
                       
    def reject(self):      
        worst_arm = self.worst()
        self.rejected[worst_arm] = True
        self.actions = np.where(~self.rejected)[0]
        
    def worst(self):
        mu = np.true_divide(self.success, self.trials)
        mu[self.rejected] = 2 # we don't want to reject the worst again
        min_val = np.min(mu)
        indicies = np.where(mu == min_val)[0] # these are the arms reported as worst
        return np.random.choice(indicies) # select one at random