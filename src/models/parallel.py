import numpy as np
from numpy.random import binomial

from .base_model import BaseModel


class Parallel(BaseModel):
    """ Parallel model as described in the paper """
    def __init__(self, q, epsilon):
        super().__init__()
        """ actions are do(x_1 = 0)...do(x_N = 0),do(x_1=1)...do(N_1=1), do() """
        assert q[0] <= .5, "P(x_1 = 1) should be <= .5 to ensure worst case reward distribution can be created"
        self.q = q
        self.N = len(q) # number of X variables (parents of Y)
        self.K = 2 * self.N + 1 #number of actions
        self.pX = np.vstack((1.0-q, q))
        self.set_epsilon(epsilon)
        self.eta, self.m = self.analytic_eta()
    
    @classmethod
    def create(cls, N, m, epsilon):
        q = cls.part_balanced_q(N, m)
        return cls(q, epsilon)
        
    @classmethod
    def most_unbalanced_q(cls, N, m):
        q = np.full(N,1.0 / m, dtype=float)
        q[0:m] = 0
        return q
    
    @classmethod
    def part_balanced_q(cls, N, m):
        """ all but m of the variables have probability .5"""
        q = np.full(N, .5, dtype=float)
        q[0:m] = 0
        return q
    
    @staticmethod
    def calculate_m(qij_sorted):
        for indx, value in enumerate(qij_sorted):
            if value >= 1.0 / (indx+1):
                return indx
        return len(qij_sorted) / 2 
        
    def action_tuple(self, action):
        """ convert from action id to the tuple (varaible,value) """
        if action == 2 * self.N + 1:
            return ((None, None))
        return (action % self.N, action / self.N)
        
    def set_epsilon(self,epsilon):
        assert epsilon <= .5, "epsilon cannot exceed .5"
        self.epsilon = epsilon
        self.epsilon_minus = self.epsilon * self.q[0] / (1.0-self.q[0]) 
        self.expected_rewards = np.full(self.K, .5)
        self.expected_rewards[0] = .5 - self.epsilon_minus
        self.expected_rewards[self.N] = .5 + self.epsilon
        self.optimal = .5 + self.epsilon
    
    def sample(self, action):
        x = binomial(1, self.pX[1,:])
        if action != self.K - 1: # everything except the do() action
            i,j = action % self.N, action / self.N
            x[i] = j
        y = binomial(1, self.pYgivenX(x))
        return x, y
    
        
    def pYgivenX(self, x):
        if x[0] == 1:
            return .5 + self.epsilon
        return .5 - self.epsilon_minus
        
    def P(self, x):
        """ calculate vector of P_a for each action a """
        indx = np.arange(len(x))
        ps = self.pX[x, indx] #probability of P(X_i = x_i) for each i given do()
        joint = ps.prod() # probability of x given do()
        pi = np.true_divide(joint, ps) # will be nan for elements for which ps is 0 
        for j in np.where(np.isnan(pi))[0]:
            pi[j] = np.prod(ps[indx != j]) 
        pij = np.vstack((pi, pi))
        pij[1-x, indx] = 0 # now this is the probability of x given do(x_i=j)
        pij = pij.reshape((len(x) * 2,)) #flatten first N-1 will be px=0,2nd px=1
        result = np.hstack((pij, joint))
        return result
        
    def analytic_eta(self):
        eta = np.zeros(self.K)
        eta[-1] =.5
        probs = self.pX[:,:].reshape((self.N * 2,)) # reshape such that first N are do(Xi=0)
        sort_order = np.argsort(probs)
        ordered = probs[sort_order]
        mq = Parallel.calculate_m(ordered)
        unbalanced = sort_order[0:mq]
        eta[unbalanced] = 1.0 / (2 * mq)
        return eta,mq