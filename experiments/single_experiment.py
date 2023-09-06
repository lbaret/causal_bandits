from abc import abstractmethod
from typing import Iterable, List, Tuple

import numpy as np


class SingleExperiment(object):
    def __init__(self, *, N: int=None, epsilon: float=None, simulations: int=None,
                 algorithms: List[Algorithm], T_vals: Iterable[int], m: int=None, model: Algorithm=None,
                 N1_vals: Iterable[int]=None, pz: float=None, q=Tuple[float, float, float, float],
                 T: int=None, pY: np.ndarray=None, N0: int=None, N1: int=None, N2: int=None) -> None:
        self.N = N
        self.epsilon = epsilon
        self.simulations = simulations
        self.algorithms = algorithms
        self.T_vals = T_vals
        self.m = m
        self.model = model
        self.N1_vals = N1_vals
        self.pz = pz
        self.q = q
        self.T = T
        self.pY = pY
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2

    def initialize_missing_attributes(self) -> None:
        return None

    @abstractmethod
    def run(self) -> None:
        ...