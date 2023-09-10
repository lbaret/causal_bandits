from typing import Callable

from experiments import (run_experiment_1, run_experiment_2, run_experiment_3,
                         run_experiment_4, run_experiment_5, run_experiment_6)

from .config.experiment_config import ExperimentConfig


# TODO : Check all possibles return types for the run method
class ExperimentRunner(object):
    def __init__(self, experiment_number: int, experiment_config: ExperimentConfig, verbose: bool=False) -> None:
        self.experiment_number = experiment_number
        self.experiment_config = experiment_config
        self.verbose = verbose

    def run(self):
        match self.experiment_number:
            case 1:
                return run_experiment_1(
                    N=self.experiment_config.N,
                    epsilon=self.experiment_config.epsilon,
                    simulations=self.experiment_config.simulations,
                    T=self.experiment_config.T,
                    verbose=self.verbose
                )
            case 2:
                return run_experiment_2(
                    N=self.experiment_config.N,
                    simulations=self.experiment_config.simulations,
                    a=self.experiment_config.a,
                    m=self.experiment_config.m,
                    verbose=self.verbose
                )
            case 3:
                return run_experiment_3(
                    N=self.experiment_config.N,
                    m=self.experiment_config.m,
                    epsilon=self.experiment_config.epsilon,
                    simulations=self.experiment_config.simulations,
                    verbose=self.verbose
                )
            case 4:
                return run_experiment_4(
                    N=self.experiment_config.N,
                    pz=self.experiment_config.pz,
                    q=self.experiment_config.q,
                    epsilon=self.experiment_config.epsilon,
                    T=self.experiment_config.T,
                    simulations=self.experiment_config.simulations,
                    verbose=self.verbose
                )
            case 5:
                return run_experiment_5(
                    N0=self.experiment_config.N0,
                    N1=self.experiment_config.N1,
                    N2=self.experiment_config.N2,
                    q=self.experiment_config.q,
                    pz=self.experiment_config.pz,
                    simulations=self.experiment_config.simulations,
                    verbose=self.verbose
                )
            case 6:
                return run_experiment_6(
                    N=self.experiment_config.N,
                    N1=self.experiment_config.N1,
                    pz=self.experiment_config.pz,
                    q=self.experiment_config.q,
                    epsilon=self.experiment_config.epsilon,
                    simulations=self.experiment_config.simulations,
                    verbose=self.verbose
                )