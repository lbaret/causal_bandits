from abc import abstractmethod
from typing import Iterable, List, Tuple

import numpy as np


class SingleExperiment(object):
    def __init__(self, *, N: int=None, epsilon: float=None, simulations: int=None,
                 algorithms: List[Algorithm], T_vals: Iterable[int], m: int=None, 
                 a: float=None, N1_vals: Iterable[int]=None, pz: float=None,
                 q=Tuple[float, float, float, float], T: int=None, pY: np.ndarray=None, 
                 N0: int=None, N1: int=None, N2: int=None, verbose: bool=False) -> None:
        self.N = N
        self.epsilon = epsilon
        self.simulations = simulations
        self.algorithms = algorithms
        self.T_vals = T_vals
        self.m = m
        self.a = a
        self.N1_vals = N1_vals
        self.pz = pz
        self.q = q
        self.T = T
        self.pY = pY
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.verbose = verbose

    def initialize_missing_attributes(self, **kwargs) -> None:
        for key, val in kwargs.items:
            setattr(self, key, val)

    @abstractmethod
    def run(self) -> None:
        ...