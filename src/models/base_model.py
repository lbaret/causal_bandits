from itertools import product

import numpy as np
from numpy.random import binomial
from scipy.optimize import minimize


class BaseModel(object):
                 
    def _expected_Y(self):
        """ Calculate the expected value of Y (over x sampled from p(x|a)) for each action """
        return np.dot(self.PY, self.A)
        
    def set_action_costs(self, costs):
        """ 
        update expected rewards to to account for action costs.
        costs should be an array of length K specifying the cost for each action.
        The expcted reward is E[Y|a] - cost(a). 
        If no costs are specified they are assume zero for all actions.
        """
        self.costs = costs
        self.expected_rewards = self.expected_Y - costs
        assert max(self.expected_rewards) <= 1
        assert min(self.expected_rewards) >= 0
        
    def make_ith_arm_epsilon_best(self ,epsilon, i):
        """ adjusts the costs such that all arms have expected reward .5, expect the first one which has reward .5 + epsilon """
        costs = self.expected_Y - 0.5
        costs[i] -= epsilon
        self.set_action_costs(costs)
        
    def pre_compute(self, compute_py: bool=True):
        """ 
        pre-computes expensive results 
        A is an lxk matrix such that A[i,j] = P(ith assignment | jth action)
        PY is an lx1 vector such that PY[i] = P(Y|ith assignment)
        """

        self.get_parent_assignments()
 
        A = np.zeros((len(self.parent_assignments), self.K))
        if compute_py:
            self.PY = np.zeros(len(self.parent_assignments))
        
        for indx,x in enumerate(self.parent_assignments):
            A[indx,:] = self.P(x)
            if compute_py:
                self.PY[indx] = self.pYgivenX(x)
            
        self.A = A
        self.A2T = (self.A**2).T
        
        self.expected_Y = self._expected_Y()
        self.expected_rewards = self.expected_Y
        
        self.eta, self.m = self.find_eta()
        self.eta = self.eta / self.eta.sum() # random choice demands more accuracy than contraint in minimizer
        
    def get_costs(self):
        if not hasattr(self, "costs"):
            self.costs = np.zeros(self.K)
        return self.costs
        
    def get_parent_assignments(self):
        if not hasattr(self, "parent_assignments") or self.parent_assignments is None:
            self.parent_assignments = BaseModel.generate_binary_assignments(self.N)
        return self.parent_assignments
    
    @classmethod
    def generate_binary_assignments(cls, N):
        """ generate all possible binary assignments to the N parents of Y. """
        return map(np.asarray, product([0,1], repeat=N))
        
    def R(self, pa, eta):
        """ returns the ratio of the probability of the given assignment under each action to the probability under the eta weighted sum of actions. """
        Q = (eta * pa).sum()
        ratio = np.true_divide(pa,Q)
        ratio[np.isnan(ratio)] = 0 # we get nan when 0/0 but should just be 0 in this case
        return ratio
                  
    def V(self, eta):
        """ returns a vector of length K with the expected value of R (over x sampled from p(x|a)) for each action a """
        #with np.errstate(divide='ignore'):        
        u = np.true_divide(1.0, np.dot(self.A, eta))
        u = np.nan_to_num(u) # converts infinities to very large numbers such that multiplying by 0 gives 0
        v = np.dot(self.A2T, u)
        return v
        
    def m_eta(self, eta):
        """ The maximum value of V"""
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), f"m should not be nan, \n{eta}\n{V}"
        return maxV
        
    def random_eta(self):
        eta = np.random.random(self.K)
        return eta / eta.sum()
        
    def _minimize(self, tol, options):
        eta0 = self.random_eta()
        constraints=({'type': 'eq', 'fun': lambda eta: eta.sum()-1.0})
        #options={'disp': True}
        res = minimize(self.m_eta, eta0, bounds=[(0.0,1.0)] * self.K, constraints=constraints, method='SLSQP', options=options)
        return res
        
    def find_eta(self,tol = 1e-10,min_starts = 1, max_starts = 10,  options={'disp': True, 'maxiter':200}):
        m = self.K + 1
        eta = None
        starts = 0
        success = 0
        while success < min_starts and starts < max_starts:
            res = self._minimize(tol, options)            
            if res.success and res.fun <= self.K:
                success +=1
                if res.fun < m:
                    m = res.fun
                    eta = res.x
            starts +=1
        
        if eta is None:
            raise Exception("optimisation failed")
    
        return eta, m
             
    def sample_multiple(self, actions, n):
        """ draws n samples from the reward distributions of the specified actions. """
        return binomial(n, self.expected_rewards[actions])
    
    def prod_all_but_j(self, vector) -> np.ndarray:
        """ returns a vector where the jth term is the product of all the entries except the jth one """
        zeros = np.where(vector==0)[0]
        if len(zeros) > 1:
            return np.zeros(len(vector))
        if len(zeros) == 1:
            result = np.zeros(len(vector))
            j = zeros[0]
            result[j] = np.prod(vector[np.arange(len(vector)) != j])
            return result

        joint = np.prod(vector)
        return np.true_divide(joint, vector)