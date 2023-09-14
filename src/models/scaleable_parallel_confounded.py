from itertools import product

import numpy as np
from numpy.random import binomial
from scipy.optimize import minimize
from scipy.special import comb

from .base_model import BaseModel


class ScaleableParallelConfounded(BaseModel):
    """ Makes use of symetries to avoid exponential combinatorics in calculating V """
    def __init__(self, q, pZ, pY, N1, N2, compute_m: bool=True) -> None:  
        super().__init__()     
        self._init_pre_action(q, pZ, pY, N1, N2, compute_m=compute_m, num_nonx_actions=3)
            
    def _init_pre_action(self, q, pZ, pY, N1, N2, compute_m: bool, num_nonx_actions: int):
        q10, q11, q20, q21 = q
        self.N = N1 + N2
        self.indx = np.arange(self.N)
        self.N1, self.N2 = N1, N2
        self.q = q
        self.pZ = pZ
        self.pytable = pY
        self.pZgivenA = np.hstack((np.full(4, pZ), 0, 1, pZ))
        pXgivenZ0 = np.hstack((np.full(N1, q10), np.full(N2, q20)))
        pXgivenZ1 = np.hstack((np.full(N1, q11), np.full(N2, q21)))
        pX0 = np.vstack((1.0-pXgivenZ0, pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        pX1 = np.vstack((1.0-pXgivenZ1, pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        self.pXgivenZ = np.stack((pX0, pX1), axis=2) # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
        self.K = 2 * self.N + num_nonx_actions
        self.qz0 = np.asarray([(1-q10), q10, (1-q20), q20])
        self.qz1 = np.asarray([(1-q11), q11, (1-q21), q21])
        self._compute_expected_reward()
        
        if compute_m:
            self.compute_m()
            
    def compute_m(self, eta_short=None):
        if eta_short is not None:
            self.m = max(self.V_short(eta_short))
            self.eta = self.expand(eta_short)
        else:
            self.eta, self.m = self.find_eta()
            
    def pYgivenX(self, x):
        i, j = x[0], x[self.N-1]
        return self.pytable[i, j]
        
    def _compute_expected_reward(self):
        q10, q11, q20, q21 = self.q
        pz = self.pZ
        a, b, c, d = self.pytable[0, 0], self.pytable[0, 1], self.pytable[1, 0], self.pytable[1, 1]
        alpha = (1-pz) * (1-q10) * (1-q20) + pz * (1-q11) * (1-q21)
        beta = (1-pz) * (1-q10) * q20 + pz * (1-q11) * q21
        gamma = (1-pz) * q10 * (1-q20) + pz * q11 * (1-q21)
        delta = (1-pz) * q10 * q20 + pz * q11 * q21
        dox10 = a * ((1-pz) * (1-q20) + pz * (1-q21)) + b * ((1-pz) * q20 + pz * q21)
        dox11 = c * ((1-pz) * (1-q20) + pz * (1-q21)) + d * ((1-pz) * q20 + pz * q21)
        dox20 = a * ((1-pz) * (1-q10) + pz * (1-q11)) + c * ((1-pz) * q10 + pz * q11)
        dox21 = b * ((1-pz) * (1-q10) + pz * (1-q11)) + d * ((1-pz) * q10 + pz * q11)
        doxj = a * alpha + b * beta + c * gamma + d * delta
        doz0 = a * (1-q10) * (1-q20) + b * (1-q10) * q20 + c * q10 * (1-q20) + d * q10 * q20
        doz1 = a * (1-q11) * (1-q21) + b * (1-q11) * q21 + c * q11 * (1-q21) + d * q11 * q21
        self.expected_Y = np.hstack((dox10, np.full(self.N-2, doxj), dox20, dox11, np.full(self.N-2, doxj), dox21, doz0, doz1, doxj))
        self.expected_rewards = self.expected_Y
        
    def P(self, x):
        n1, n2 = x[0:self.N1].sum(), x[self.N1:].sum()
        pz0, pz1 = self.p_n_given_z(n1, n2)
        pc = self.pZgivenA * pz1 + (1-self.pZgivenA) * pz0
        doxi0 = np.hstack((np.full(self.N1, pc[0]), np.full(self.N2, pc[1])))
        doxi1 = np.hstack((np.full(self.N1, pc[2]), np.full(self.N2, pc[3])))
        pij = np.vstack((doxi0, doxi1))
        pij[1-x, self.indx] = 0
        pij = pij.reshape((self.N * 2,))
        result = np.hstack((pij, pc[4], pc[5], pc[6]))
        return result
        
    def sample(self, action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        if action == 2 * self.N+1: # do(z = 1)
            z = 1       
        elif action == 2 * self.N: # do(z = 0)
            z = 0     
        else: # we are not setting z
            z = binomial(1, self.pZ)
        
        x = binomial(1, self.pXgivenZ[1, :, z]) # PXgivenZ[j,i,k] = P(X_i=j|Z=k)
        
        if action < 2 * self.N: # setting x_i = j
             i, j = action % self.N, action / self.N
             x[i] = j
             
        y = binomial(1, self.pYgivenX(x)) 
        
        return x, y
        
        
    def V_short(self, eta):
        sum0 = np.zeros(7, dtype=float)
        sum1 = np.zeros(7, dtype=float)
        for n1, n2 in product(range(self.N1+1), range(self.N2+1)):
             wdo = comb(self.N1, n1, exact=True) * comb(self.N2, n2, exact=True)
             wdox10 = comb(self.N1-1, n1, exact=True) * comb(self.N2, n2, exact=True)
             wdox11 = comb(self.N1-1, n1-1, exact=True) * comb(self.N2, n2, exact=True)
             wdox20 = comb(self.N1, n1, exact=True) * comb(self.N2-1, n2, exact=True)
             wdox21 = comb(self.N1, n1, exact=True) * comb(self.N2-1, n2-1, exact=True)
             w = np.asarray([wdox10, wdox20, wdox11, wdox21, wdo, wdo, wdo])
             
             pz0, pz1 = self.p_n_given_z(n1, n2)

             counts = [self.N1-n1, self.N2-n2, n1, n2, 1, 1, 1]
             Q = (eta * pz0 * counts * (1-self.pZgivenA) + eta * pz1 * counts * self.pZgivenA).sum()
             
             ratio = np.nan_to_num(np.true_divide(pz0 * (1-self.pZgivenA) + pz1 * self.pZgivenA, Q))
          
             sum0 += np.asfarray(w * pz0 * ratio)
             sum1 += np.asfarray(w * pz1 * ratio)
        result = self.pZgivenA * sum1 + (1-self.pZgivenA) * sum0
        return result
        
    def V(self, eta):
        eta_short_form = self.contract(eta)
        v = self.V_short(eta_short_form)
        v_long = self.expand(v)
        return v_long
        
    def m_rep(self, eta_short_form):
        V = self.V_short(eta_short_form)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
    
    def find_eta(self, tol: float=1e-10):
        eta, m = super().find_eta()
        self.eta_short = eta
        eta_full = self.expand(eta)
        return eta_full, m 
        
    def _minimize(self, tol: float, options):
        weights = self.weights()
        eta0 = self.random_eta_short()
        constraints=({'type': 'eq', 'fun': lambda eta: np.dot(eta, weights)-1.0})
        res = minimize(self.m_rep, eta0, bounds=[(0.0,1.0)] * len(eta0), constraints=constraints, method='SLSQP', tol=tol, options=options)      
        return res
    
    def weights(self):
        return np.asarray([self.N1, self.N2, self.N1, self.N2, 1, 1, 1])
                   
            
    def p_n_given_z(self, n1, n2):
        powers = np.tile([self.N1-n1, n1, self.N2-n2, n2], 7).reshape((7, 4))
        powers[0, 0]-= 1 #do(x1=0)
        powers[1, 2]-= 1 #do(x2=0)
        powers[2, 1]-= 1 #do(x1=1)
        powers[3, 3]-= 1 #do(x2=1)
        
        pnz0 = (self.qz0**powers).prod(axis=1)
        pnz1 = (self.qz1**powers).prod(axis=1)
        return pnz0, pnz1
        
    def random_eta_short(self):
        weights = self.weights()
        eta0 = np.random.random(len(weights))
        eta0 = eta0 / np.dot(weights, eta0)
        return eta0
        
    def contract(self, long_form):
        result = np.zeros(7)
        result[0] = long_form[0]
        result[1] = long_form[self.N-1]
        result[2] = long_form[self.N]
        result[3] = long_form[2 * self.N-1]
        result[4:] = long_form[-3:]
        return result
        
    def expand(self, short_form):
        arrays = []
        for indx, count in enumerate(self.weights()):
            arrays.append(np.full(count, short_form[indx]))
        result = np.hstack(arrays)
        return result