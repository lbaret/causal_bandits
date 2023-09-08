from typing import Iterable, List, Tuple

import numpy as np

from experiments.single_experiment import SingleExperiment
from src.models import Parallel


class RegretVsT(SingleExperiment):
    def __init__(self, *, N: int = None, epsilon: float = None, simulations: int = None, algorithms: List, m: int = None, verbose: bool = False) -> None:
        super().__init__(N=N, epsilon=epsilon, simulations=simulations, algorithms=algorithms, m=m, verbose=verbose)

        model = Parallel.create(self.N, self.m, self.epsilon)
        T_vals = range(10, 6 * model.K, 25)

        self.initialize_missing_attributes(
            model=model,
            T_vals=T_vals
        )

    def run(self) -> None:
        regret = np.zeros((len(self.algorithms), len(self.T_vals), self.simulations))
        pulls = np.zeros((len(self.algorithms), len(self.T_vals), self.model.K), dtype=int)
        for T_indx, T in enumerate(self.T_vals): 
            for a_indx, algorithm in enumerate(self.algorithms):
                for s in range(self.simulations):
                    regret[a_indx, T_indx, s] = algorithm.run(T, self.model)
                    if algorithm.best_action is not None:
                        pulls[a_indx, T_indx, algorithm.best_action] += 1
            print(T)
                    
        return regret, pulls