from math import ceil, sqrt

import numpy as np

from experiments.experiment_config import ExperimentConfig, now_string
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import Parallel


class Experiment(object):
    def __init__(self) -> None:
        pass
    
    def run(self) -> None:
        pass

    def __call__(self) -> None:
        self.run()

# =====================================================
experiment = ExperimentConfig(1)
experiment.log_code()

N = 50
epsilon = .3
simulations = 100
T = 400
algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]
m_vals = range(2,N,2)
    
regret,models = regret_vs_m(algorithms,m_vals,N,T,epsilon,simulations = simulations)

experiment.plot_regret(regret,m_vals,"m",algorithms,legend_loc="lower right")

# =====================================================
experiment = ExperimentConfig(2)
experiment.log_code()

N= 50
simulations = 10000
a = 9.0
m = 2
model = Parallel.create(N,m,.1)

Tmin = int(ceil(4*model.K/a))
Tmax = 10*model.K
T_vals = range(Tmin,Tmax,100)

algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret = regret_vs_T_vary_epsilon(model,algorithms,T_vals,simulations = simulations)
finished = now_string()

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)

# =====================================================
experiment = ExperimentConfig(3)
experiment.log_code()
                  
simulations = 10000
N = 50
m = 2
epsilon = .3
model = Parallel.create(N,m,epsilon)
T_vals = range(10,6*model.K,25)
algorithms = [GeneralCausal(truncate='None'),ParallelCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret = regret_vs_T(model,algorithms,T_vals,simulations = simulations)
finished = now_string()


experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)

# =====================================================
experiment = ExperimentConfig(4)
experiment.log_code()
    
N = 50
N1_vals = range(1,N,3)
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .3
simulations = 10000
T = 400
algorithms = [SuccessiveRejects(),GeneralCausal(),AlphaUCB(2),ThompsonSampling()]


epsilon = .3
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

m_vals,regret,models = regret_vs_m_general(algorithms,N1_vals,N,T,pz,pY,q,epsilon,simulations = simulations)
experiment.plot_regret(regret,m_vals,"m",algorithms,legend_loc = "lower right",legend_extra = [ParallelCausal])

# =====================================================
experiment = ExperimentConfig(5)
experiment.log_code()
           
simulations = 10000                 

q = (.2,.8,.7,.3)
pz = 0.6 
pY = np.asarray([[ 0 , 1  ],[1 , 0]])

N0 = 6
N1 = 1
N2 = 14
N = N0+N1+N2

q10,q11,q20,q21 = q
pXgivenZ0 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q10),np.full(N2,q20)))    
pXgivenZ1 = np.hstack((np.full(N0,1.0/N0),np.full(N1,q11),np.full(N2,q21)))
pXgivenZ = np.stack((np.vstack((1.0-pXgivenZ0,pXgivenZ0)),np.vstack((1.0-pXgivenZ1,pXgivenZ1))),axis=2) # PXgivenZ[i,j,k] = P(X_j=i|Z=k)
pYfunc = lambda x: pY[x[N0],x[N-1]]
model = ParallelConfoundedNoZAction(pz,pXgivenZ,pYfunc)

T_vals = list(chain(range(10,100,25),range(85+50,450,50),range(435+100,1036,100)))

algorithms = [GeneralCausal(),ParallelCausal(),SuccessiveRejects(),ThompsonSampling(),AlphaUCB(2)]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)

print("m",model.m)

# =====================================================
experiment = ExperimentConfig(6)
experiment.log_code()
                
N = 50
N1 = 1
pz = .4
q = (0.00001,0.00001,.4,.65)
epsilon = .3
pY = ParallelConfounded.pY_epsilon_best(q,pz,epsilon)

simulations = 10000

model = ScaleableParallelConfounded(q,pz,pY,N1,N-N1)

T_vals = range(25,626,25)

algorithms = [GeneralCausal(),SuccessiveRejects(),AlphaUCB(2),ThompsonSampling()]

regret,pulls = regret_vs_T(model,algorithms,T_vals,simulations = simulations)

experiment.plot_regret(regret,T_vals,"T",algorithms,legend_loc = None)