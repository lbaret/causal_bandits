from typing import Tuple


class ExperimentConfig(object):
    def __init__(self, *, N: int=None, N0: int=None, N1: int=None, N2: int=None, epsilon: float=None, simulations: int=None, 
                 T: int=None, a: float=None, m:int=None, pz:float=None, q: Tuple[float, float, float, float]=None) -> None:
        self.N = N
        self.N0 = N0
        self.N1 = N1
        self.N2 = N2
        self.epsilon = epsilon
        self.simulations = simulations
        self.T = T
        self.a = a
        self.m = m
        self.pz = pz
        self.q = q