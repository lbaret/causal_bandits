from typing import Iterable, List, Tuple

import numpy as np

from src.models import Parallel

from .single_experiment import SingleExperiment


class RegretVsM(SingleExperiment):
    def __init__(self, *, N: int = None, epsilon: float = None, simulations: int = None, algorithms: List, verbose: bool=False) -> None:
        super().__init__(N=N, epsilon=epsilon, simulations=simulations, algorithms=algorithms, verbose=verbose)
        
        self.initialize_missing_attributes(
            m_vals=range(2, self.N, 2)
        )

    def run(self) -> None:
        models = []
        regret = np.zeros((len(self.algorithms), len(self.m_vals), self.simulations))
        for m_indx, m in enumerate(self.m_vals):
            model = Parallel.create(self.N, m, self.epsilon)
            models.append(model)
            
            if self.verbose:
                print("built model {0}".format(m))
            
            for s in range(self.simulations):
                for a_indx, algorithm in enumerate(self.algorithms):
                    regret[a_indx, m_indx, s] = algorithm.run(self.T, model)
        
        return regret, models