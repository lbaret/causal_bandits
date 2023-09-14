import numpy as np
from numpy.random import binomial

from .scaleable_parallel_confounded import ScaleableParallelConfounded


class ScaleableParallelConfoundedNoZAction(ScaleableParallelConfounded):
      
    def __init__(self, q, pZ, pY, N1, N2, compute_m=True) -> None:
        super().__init__(q, pZ, pY, N1, N2, compute_m)
        self._init_pre_action(q, pZ, pY, N1, N2, compute_m, 1)
        self.expected_rewards = self._mask(self.expected_rewards)
        self.expected_Y = self._mask(self.expected_Y)

    def _mask(self, vect):
        return np.hstack((vect[0:-3], vect[-1]))
        
    def P(self, x):
        p = super().P(x)
        return self._mask(p)
    
    def V(self, eta):
        eta_short_form = self.contract(eta)
        eta = np.hstack((eta_short_form[0:-1], 0, 0, eta_short_form[-1]))
        v = self.V_short(eta) # length 7
        v = self._mask(v) # length 5
        v_long = self.expand(v)
        return v_long
        
    def sample(self, action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        z = binomial(1, self.pZ)        
        x = binomial(1, self.pXgivenZ[1,:,z]) # PXgivenZ[j,i,k] = P(X_i=j|Z=k)
        
        if action < 2 * self.N: # setting x_i = j
             i,j = action % self.N, action / self.N
             x[i] = j
             
        y = binomial(1, self.pYgivenX(x)) 
        
        return x, y
    
    def weights(self) -> np.ndarray:
        return np.asarray([self.N1, self.N2, self.N1, self.N2, 1])   
        
    def m_rep(self, eta_short_form):
        eta = np.hstack((eta_short_form[0:-1], 0, 0, eta_short_form[-1]))
        V = self.V_short(eta)
        V[-3:-1] = 0 # exclude do(z=0) and do(z=1)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def contract(self, long_form):
        result = np.zeros(5)
        result[0] = long_form[0]
        result[1] = long_form[self.N-1]
        result[2] = long_form[self.N]
        result[3] = long_form[2 * self.N-1]
        result[4] = long_form[-1]
        return result