import numpy as np
from numpy.random import binomial

from .parallel_confounded import ParallelConfounded


class ParallelConfoundedNoZAction(ParallelConfounded):
    """ the ParallelConfounded Model but without the actions that set Z """
    def __init__(self, pZ, pXgivenZ, pYfunc):
        super().__init__(pZ, pXgivenZ, pYfunc)
        self._init_pre_action(pZ, pXgivenZ, pYfunc, 1)
        self.pre_compute()
              
    def P(self, x):
        p = super().P(x)
        return np.hstack((p[0:-3], p[-1]))        
        
    def sample(self, action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        z = binomial(1, self.pZ)        
        x = binomial(1, self.pXgivenZ[1, :, z]) # PXgivenZ[j,i,k] = P(X_i=j|Z=k)
        
        if action < 2 * self.N: # setting x_i = j
             i,j = action % self.N, action / self.N
             x[i] = j
             
        y = binomial(1, self.pYgivenX(x)) 
        
        return x, y