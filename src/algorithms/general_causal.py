from math import log, sqrt

import numpy as np

from .base_algorithm import BaseAlgorithm


class GeneralCausal(BaseAlgorithm):
    label = "Algorithm 2"
    
    def __init__(self,truncate = "clip"):
        super().__init__()
        self.truncate = truncate
        #self.label = "Algorithm 2-"+truncate

    def run(self, T, model):
        eta = model.eta
        m = model.m
        n = len(eta)
        self.B = sqrt(m * T / log(2.0 * T * n))
        
        actions = range(n)
        u = np.zeros(n)
        for t in range(T):
            a = np.random.choice(actions, p=eta)
            x, y = model.sample(a) #x is an array containing values for each variable
            #y = y - model.get_costs()[a]
            pa = model.P(x)
            r = model.R(pa, eta)
            if self.truncate == "zero":
                z = (r <= self.B) * r * y
            elif self.truncate == "clip":
                z = np.minimum(r, self.B) * y
            else:
                z = r * y
                
            u += z
        self.u = u / float(T)
        r = self.u - model.get_costs()
        self.best_action = np.argmax(r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action] 