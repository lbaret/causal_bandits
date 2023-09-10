# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:30:34 2016

@author: finn
"""
from math import ceil, sqrt
from typing import Iterable

import numpy as np

from experiments.experiment_config import ExperimentConfig, now_string
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import Parallel


def regret_vs_T_vary_epsilon(model, algorithms, T_vals: Iterable, a: float, simulations: int=10, verbose: bool=False):
    regret = np.zeros((len(algorithms), len(T_vals), simulations))
    
    for T_indx, T in enumerate(T_vals):
        if verbose:
            print(T)

        epsilon = sqrt(model.K / (a * T))
        model.set_epsilon(epsilon)

        for s in range(simulations):
            if verbose:
                print(f"\rSimulation : {s} / {simulations}", end='')

            for a_indx, algorithm in enumerate(algorithms):
                regret[a_indx, T_indx, s] = algorithm.run(T, model)
    
        if verbose:
            print()

    return regret

def run_experiment_2(verbose: bool=False) -> None:
    experiment = ExperimentConfig(2)
    experiment.log_code()

    N= 50
    simulations = 10000
    a = 9.0
    m = 2
    model = Parallel.create(N, m, .1)

    Tmin = int(ceil(4*model.K / a))
    Tmax = 10*model.K
    T_vals = range(Tmin, Tmax, 100)

    algorithms = [GeneralCausal(truncate='None'), ParallelCausal(), SuccessiveRejects(), AlphaUCB(2), ThompsonSampling()]

    regret = regret_vs_T_vary_epsilon(model, algorithms, T_vals, simulations, verbose)
    finished = now_string()

    experiment.plot_regret(regret, T_vals, "T", algorithms, legend_loc=None)




