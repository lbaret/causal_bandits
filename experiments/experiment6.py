# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:19:07 2016

@author: finn
"""

from typing import Iterable, Tuple

import numpy as np

from experiments.config.experiment_factory import ExperimentFactory
from src.algorithms import (AlphaUCB, GeneralCausal, SuccessiveRejects,
                            ThompsonSampling)
from src.models import ParallelConfounded, ScaleableParallelConfounded


def regret_vs_T(model, algorithms, T_vals: Iterable,simulations: int=10, verbose: bool=False):
    regret = np.zeros((len(algorithms), len(T_vals), simulations))
    pulls = np.zeros((len(algorithms), len(T_vals), model.K), dtype=int)
    for T_indx, T in enumerate(T_vals):
        if verbose:
            print(f"{T}")
        for a_indx,algorithm in enumerate(algorithms):
            for s in range(simulations):
                if verbose:
                    print(f"\r{s} / {simulations}", end='')

                regret[a_indx, T_indx, s] = algorithm.run(T, model)

                if algorithm.best_action is not None:
                    pulls[a_indx, T_indx, algorithm.best_action] += 1
            
            if verbose:
                print()
        if verbose:
            print(T)
                
    return regret,pulls
           

def run_experiment_6(N: int, N1: int, pz: float, q: Tuple[float, float, float, float], 
                     epsilon: float, simulations: int, verbose: bool=False) -> None:
    experiment = ExperimentFactory(6)
    experiment.log_code()

    pY = ParallelConfounded.pY_epsilon_best(q, pz, epsilon)

    model = ScaleableParallelConfounded(q, pz, pY, N1, N - N1)
    T_vals = range(25, 626, 25)

    algorithms = [GeneralCausal(), SuccessiveRejects(), AlphaUCB(2), ThompsonSampling()]

    regret, pulls = regret_vs_T(model, algorithms, T_vals, simulations, verbose)

    experiment.plot_regret(regret, T_vals, "T", algorithms, legend_loc=None)
