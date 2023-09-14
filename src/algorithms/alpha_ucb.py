import numpy as np

from .ucb import UCB


class AlphaUCB(UCB):
    """ Implementation based on ... """
    label = "UCB"
    
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
    
    def upper_bound(self, t):
        mu = np.true_divide(self.success,self.trials)
        interval = np.sqrt(self.alpha * np.log(t) / (2.0 * self.trials))
        return mu + interval