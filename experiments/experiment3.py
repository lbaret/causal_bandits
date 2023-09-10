# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:52:59 2016

@author: finn
"""
from typing import Iterable
import numpy as np

from experiments.config.experiment_factory import ExperimentFactory, now_string
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import Parallel


def regret_vs_T(model, algorithms, T_vals: Iterable, simulations: int=10, verbose: bool=False):
    regret = np.zeros((len(algorithms), len(T_vals), simulations))
    
    for T_indx, T in enumerate(T_vals):
        if verbose:
            print(T)

        for a_indx, algorithm in enumerate(algorithms):
            for s in range(simulations):
                if verbose:
                    print(f"{s} / {simulations}", end='')
                regret[a_indx, T_indx, s] = algorithm.run(T, model)
        if verbose:
            print()
                
    return regret

def run_experiment_3(N: int, m: int, epsilon: float, simulations: int, verbose: bool=False) -> None:
    experiment = ExperimentFactory(3)
    experiment.log_code()

    model = Parallel.create(N, m, epsilon)
    T_vals = range(10, 6 * model.K, 25)
    algorithms = [GeneralCausal(truncate='None'), ParallelCausal(), SuccessiveRejects(), AlphaUCB(2), ThompsonSampling()]

    regret = regret_vs_T(model, algorithms, T_vals, simulations, verbose)
    finished = now_string()


    experiment.plot_regret(regret, T_vals, "T", algorithms, legend_loc=None)



