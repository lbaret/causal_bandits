# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:47:47 2016

@author: finn
"""
from typing import Iterable
import numpy as np

from experiments.config.experiment_factory import ExperimentFactory
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import Parallel


def regret_vs_m(algorithms, m_vals: Iterable, N: int, T: int, epsilon: float, 
                simulations: int=10, verbose: bool=False):
    models = []    
    regret = np.zeros((len(algorithms), len(m_vals), simulations))

    for m_indx, m in enumerate(m_vals):
        if verbose:
            print(m)
        model = Parallel.create(N, m, epsilon)
        models.append(model)

        if verbose:
            print(f"built model {m}")

        for s in range(simulations):
            if verbose:
                print(f"\rSimulation : {s} / {simulations}", end='')
            
            for a_indx, algorithm in enumerate(algorithms):
                regret[a_indx, m_indx, s] = algorithm.run(T, model)
        
        if verbose:
            print()
    
    return regret, models

def run_experiment_1(N: int, epsilon: float, simulations: int, T: int, verbose: bool=False) -> None:
    experiment = ExperimentFactory(1)
    experiment.log_code()
    
    algorithms = [GeneralCausal(truncate='None'), ParallelCausal(), SuccessiveRejects(), AlphaUCB(2), ThompsonSampling()]
    m_vals = range(2, N, 2)
        
    regret, _ = regret_vs_m(algorithms, m_vals, N, T, epsilon, simulations, verbose)

    experiment.plot_regret(regret, m_vals, "m", algorithms, legend_loc="lower right")



    
