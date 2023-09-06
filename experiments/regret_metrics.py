from math import ceil, sqrt

import numpy as np

from experiments.experiment_config import ExperimentConfig, now_string
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import Parallel


# TODO : Run debug and check each parameters types
class RegretMetrics(object):
    @staticmethod
    def regret_vs_m(algorithms, m_vals, N, T, epsilon, simulations=10):  
        models = []    
        regret = np.zeros((len(algorithms),len(m_vals),simulations))
        for m_indx, m in enumerate(m_vals):
            model = Parallel.create(N, m, epsilon)
            models.append(model)
            print("built model {0}".format(m))
            for s in range(simulations):
                for a_indx, algorithm in enumerate(algorithms):
                    regret[a_indx, m_indx, s] = algorithm.run(T, model)
        
        return regret, models
    
    @staticmethod
    def regret_vs_T_vary_epsilon(model, algorithms, T_vals, a, simulations=10):
        regret = np.zeros((len(algorithms), len(T_vals), simulations))
        
        for T_indx, T in enumerate(T_vals): 
            print(T)
            epsilon = sqrt(model.K / (a*T))
            model.set_epsilon(epsilon)
            for s in range(simulations):
                for a_indx, algorithm in enumerate(algorithms):
                    regret[a_indx, T_indx, s] = algorithm.run(T, model)
            
        return regret
    
    # @staticmethod
    # def regret_vs_T(model, algorithms, T_vals, simulations=10):
    #     regret = np.zeros((len(algorithms), len(T_vals), simulations))
        
    #     for T_indx, T in enumerate(T_vals): 
    #         for a_indx, algorithm in enumerate(algorithms):
    #             for s in range(simulations):
    #                 regret[a_indx, T_indx, s] = algorithm.run(T, model)
    #         print(T)
                    
    #     return regret
    
    @staticmethod
    def regret_vs_T(model, algorithms, T_vals, simulations=10):
        regret = np.zeros((len(algorithms), len(T_vals), simulations))
        pulls = np.zeros((len(algorithms), len(T_vals), model.K), dtype=int)
        for T_indx, T in enumerate(T_vals): 
            for a_indx, algorithm in enumerate(algorithms):
                for s in range(simulations):
                    regret[a_indx, T_indx, s] = algorithm.run(T, model)
                    if algorithm.best_action is not None:
                        pulls[a_indx, T_indx, algorithm.best_action] += 1
            print(T)
                    
        return regret, pulls
    
    @staticmethod
    def regret_vs_m_general(algorithms, N1_vals, N, T, pz, pY, q, epsilon, simulations=1000): 
        m_vals = []
        models = []
        regret = np.zeros((len(algorithms), len(N1_vals), simulations))

        for m_indx,N1 in enumerate(N1_vals):
            model = ScaleableParallelConfounded(q, pz, pY, N1, N-N1, compute_m = False)
            eta = [0 ,0, 1.0 / (N1 + 2.0), 0, 0, 0, 1-N1 / (N1 + 2.0)]
            model.compute_m(eta_short=eta)
        
            print(N1, model.m)
            m_vals.append(model.m)
            models.append(model)
            for a_indx, algorithm in enumerate(algorithms):
                for s in range(simulations):
                    regret[a_indx, m_indx, s] = algorithm.run(T, model)
                    
        
        return m_vals, regret, models