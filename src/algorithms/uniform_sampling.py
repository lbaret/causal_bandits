import numpy as np

from .base_algorithm import BaseAlgorithm


class UniformSampling(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        
    label = "Uniform"

    def run(self, T, model):
        trials_per_action = T / model.K
        success = model.sample_multiple(range(model.K), trials_per_action)
        self.u = np.true_divide(success, trials_per_action)
        self.best_action = self.argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]