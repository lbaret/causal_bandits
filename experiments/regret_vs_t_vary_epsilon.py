from math import sqrt, ceil
from typing import Iterable, List, Tuple

import numpy as np

from src.models import Parallel

from .single_experiment import SingleExperiment


class RegretVsTVaryEpsilon(SingleExperiment):
    def __init__(self, *, simulations: int = None, algorithms: List, m: int = None, verbose: bool = False) -> None:
        super().__init__(simulations=simulations, algorithms=algorithms, m=m, verbose=verbose)

        model = Parallel.create(self.N, self.m, .1)

        Tmin = int(ceil(4 * self.model.K / self.a))
        Tmax = 10 * self.model.K
        T_vals = range(Tmin, Tmax, 100)
        
        self.initialize_missing_attributes(
            model=model,
            T_vals=T_vals
        )

    def run(self) -> None:
        regret = np.zeros((len(self.algorithms), len(self.T_vals), self.simulations))
        
        for T_indx, T in enumerate(self.T_vals):
            if self.verbose: 
                print(T)
            
            epsilon = sqrt(self.model.K / (self.a*T))
            self.model.set_epsilon(epsilon)
            
            for s in range(self.simulations):
                for a_indx, algorithm in enumerate(self.algorithms):
                    regret[a_indx, T_indx, s] = algorithm.run(T, self.model)
            
        return regret