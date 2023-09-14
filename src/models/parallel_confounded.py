import numpy as np
from numpy.random import binomial
from scipy.optimize import minimize

from .base_model import BaseModel


class ParallelConfounded(BaseModel):
    """ Represents a parallel bandit with one common confounder. Z ->(X1 ... XN) and (X1,...,XN) -> Y 
        Actions are do(x_1 = 0),...,do(x_N = 0), do(x_1=1),...,do(x_N = 1),do(Z=0),do(Z=1),do()"""
    
    def __init__(self, pZ, pXgivenZ, pYfunc) -> None:
        super().__init__()
        self._init_pre_action(pZ, pXgivenZ, pYfunc, 3)
        self.pre_compute()
        
    def _init_pre_action(self, pZ, pXgivenZ, pYfunc, num_non_x_actions) -> None:
        """ The initialization that should occur regardless of whether we can act on Z """
        self.N = pXgivenZ.shape[1]
        self.indx = np.arange(self.N)
        self.pZ = pZ
        self.pXgivenZ = pXgivenZ # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
        self.pYfunc = pYfunc
        #self.pytable = pY #np.asarray([[.4,.4],[.7,.7]])  
        
        # variables X for which pXgivenZ is identical must have the same value for eta.
        group_values = []
        self.group_members = [] #variables in each group
        for var in range(self.N):
            matched = False
            value = self.pXgivenZ[:, var, :]
            for group, gv in enumerate(group_values):
                if np.allclose(value, gv):
                    self.group_members[group].append(var)
                    matched = True
                    break
            if not matched:
                group_values.append(value)
                self.group_members.append([var])
        counts = [len(members) for members in self.group_members]
        self.group_members = [np.asarray(members, dtype=int) for members in self.group_members]
        
        self.weights = list(chain(counts * 2,[1] * num_non_x_actions))
        self.K = 2 * self.N + num_non_x_actions
        self.nnx = num_non_x_actions
        
    @classmethod
    def pY_epsilon_best(cls,q,pZ,epsilon):
        """ returns a table pY with Y epsilon-optimal for X1=1, sub-optimal for X1=0 and .5 for all others"""
        q10, q11, q20, q21 = q
        px1 = (1-pZ) * q10 + pZ * q11
        px0 = (1-pZ) * (1-q10) + pZ * (1-q11)              
        epsilon2 = (px1 / px0) * epsilon
        assert epsilon2 < .5
        pY = np.asarray([[.5-epsilon2, .5-epsilon2], [.5+epsilon, .5+epsilon]])     
        return pY
               
    @classmethod
    def create(cls, N, N1, pz, pY, q):
        """ builds ParallelConfounded model"""
        q10, q11, q20, q21 = q
        N2 = N - N1
        pXgivenZ0 = np.hstack((np.full(N1, q10), np.full(N2, q20)))
        pXgivenZ1 = np.hstack((np.full(N1, q11), np.full(N2, q21)))
        pX0 = np.vstack((1.0-pXgivenZ0, pXgivenZ0)) # PX0[j,i] = P(X_i = j|Z = 0)
        pX1 = np.vstack((1.0-pXgivenZ1, pXgivenZ1)) # PX1[i,j] = P(X_i = j|Z = 1)
        pXgivenZ = np.stack((pX0, pX1), axis=2) # PXgivenZ[i,j,k] = P(X_i=j|Z=k)
        pYfunc = lambda x: pY[x[0], x[N-1]]
        model = cls(pz, pXgivenZ, pYfunc)
        return model
       
    def pYgivenX(self, x):
        return self.pYfunc(x)
    
    def action_tuple(self, action):
        """ convert from action id to the tuple (varaible,value) """
        if action == 2 * self.N + 1:
            return ('z', 1)
        if action ==  2 * self.N:
            return ('z', 0)
        if action == 2 * self.N + 2:
            return ((None, None))
        return (action % self.N, action / self.N)
           
    def sample(self, action):
        """ samples given the specified action index and returns the values of the parents of Y, Y. """   
        if action == 2 * self.N + 1: # do(z = 1)
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
                
    def P(self, x):
        """ calculate P(X = x|a) for each action a. 
            x is an array of length N specifiying an assignment to the parents of Y
            returns a vector of length K. 
        """
        pz1 = self.pXgivenZ[x, self.indx, 1]
        pz0 = self.pXgivenZ[x, self.indx, 0]
    
        p_obs = self.pZ * pz1.prod() + (1-self.pZ) * pz0.prod()
        
        # for do(x_i = j)
        joint_z0 = self.prod_all_but_j(pz0) # vector of length N
        joint_z1 = self.prod_all_but_j(pz1)
        p = self.pZ * joint_z1 + (1-self.pZ) * joint_z0  
        pij = np.vstack((p, p))
        pij[1-x, self.indx] = 0 # 2*N array, pij[i,j] = P(X=x|do(X_i=j)) = d(X_i-j)*prod_k!=j(X_k = x_k)
        pij = pij.reshape((len(x) * 2,)) #flatten first N-1 will be px=0,2nd px=1
        
        result = np.hstack((pij, pz0.prod(), pz1.prod(), p_obs))
        return result
        
    def _minimize(self,tol,options):
        eta0 = np.random.random(len(self.group_members) * 2 + self.nnx)
        eta0 = eta0 / np.dot(self.weights, eta0)
        
        constraints=({'type': 'eq', 'fun': lambda eta: np.dot(eta, self.weights) - 1.0})
        res = minimize(self.m_rep, eta0, bounds=[(0.0, 1.0)] * len(eta0), constraints=constraints, method='SLSQP', tol=tol, options=options)      
        return res
            
    def find_eta(self, tol: float=1e-10):
        eta, m = super().find_eta()
        self.eta_short = eta
        eta_full = self.expand(eta)
        return eta_full, m 
        
    def m_rep(self, eta_short_form):
        eta = self.expand(eta_short_form)
        V = self.V(eta)
        maxV = V.max()
        assert not np.isnan(maxV), "m must not be nan"
        return maxV
        
    def expand(self, short_form):
        eta_full = np.zeros(self.K)
        eta_full[-self.nnx:] = short_form[-self.nnx:]
        num_groups = len(self.group_members)
        for group, members in enumerate(self.group_members):
            eta0 = short_form[group]
            eta1 = short_form[num_groups+group]
            eta_full[members] = eta0
            eta_full[members+self.N] = eta1
        return eta_full