from typing import Iterable, List, Tuple
import numpy as np

from experiments.single_experiment import SingleExperiment
from src.models import ScaleableParallelConfounded

class RegretMetrics(SingleExperiment):
    def __init__(self, *, simulations: int = None, algorithms: List, N1_vals: Iterable[int] = None, pz: float = None, q=..., T: int = None, pY: ndarray = None, N1: int = None, verbose: bool = False) -> None:
        super().__init__(simulations=simulations, algorithms=algorithms, N1_vals=N1_vals, pz=pz, q=q, T=T, pY=pY, N1=N1, verbose=verbose)
        self.initialize_missing_attributes(
            m_vals=[],
            models=[],
            regret=np.zeros((len(self.algorithms), len(self.N1_vals), self.simulations)),
            eta=[0 ,0, 1.0 / (self.N1 + 2.0), 0, 0, 0, 1 - self.N1 / (self.N1 + 2.0)]
        )

    def run(self) -> None: 
        for m_indx, N1 in enumerate(self.N1_vals):
            model = ScaleableParallelConfounded(self.q, self.pz, self.pY, N1, self.N - N1, compute_m=False)
            model.compute_m(eta_short=self.eta)

            if self.verbose:
                print(N1, model.m)

            self.m_vals.append(model.m)
            self.models.append(model)
            for a_indx, algorithm in enumerate(self.algorithms):
                for s in range(self.simulations):
                    self.regret[a_indx, m_indx, s] = algorithm.run(self.T, model)