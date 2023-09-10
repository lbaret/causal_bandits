# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:48:05 2016

@author: finn


"""
from typing import Iterable, Tuple

import numpy as np

from experiments.config.experiment_factory import ExperimentFactory
from src.algorithms import (AlphaUCB, GeneralCausal, ParallelCausal,
                            SuccessiveRejects, ThompsonSampling)
from src.models import ParallelConfounded, ScaleableParallelConfounded


def regret_vs_m_general(algorithms, N1_vals: Iterable, N: int, T: int, pz: float, 
                        pY: np.ndarray, q: Tuple[float, float, float, float], 
                        epsilon: float, simulations: int=1000, verbose: bool=False): 
    m_vals = []
    models = []
    regret = np.zeros((len(algorithms), len(N1_vals), simulations))

    for m_indx, N1 in enumerate(N1_vals):
        model = ScaleableParallelConfounded(q, pz, pY, N1, N - N1, compute_m=False)
        eta = [0, 0, 1.0 / (N1 + 2.0), 0, 0, 0, 1 - N1 / (N1 + 2.0)]
        model.compute_m(eta_short=eta)

        if verbose:
            print(N1, model.m)

        m_vals.append(model.m)
        models.append(model)
        for a_indx, algorithm in enumerate(algorithms):
            for s in range(simulations):
                if verbose:
                    print(f"\r{s} / {simulations}", end='')
                regret[a_indx, m_indx, s] = algorithm.run(T,model)
        
        if verbose:
            print()

    return m_vals, regret, models
    
def run_experiment_4(N: int, pz: float, q: Tuple[float, float, float, float], epsilon: float, 
                     T: int, simulations: int, verbose: bool=False) -> None:
    experiment = ExperimentFactory(4)
    experiment.log_code()

    N1_vals = range(1, N, 3)
    algorithms = [SuccessiveRejects(), GeneralCausal(), AlphaUCB(2), ThompsonSampling()]

    pY = ParallelConfounded.pY_epsilon_best(q, pz, epsilon)

    m_vals, regret, models = regret_vs_m_general(algorithms, N1_vals, N, T, pz, pY, q, epsilon, simulations, verbose)
    experiment.plot_regret(regret, m_vals, "m", algorithms, legend_loc="lower right", legend_extra=[ParallelCausal])








