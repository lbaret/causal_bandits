import numpy as np

from .base_algorithm import BaseAlgorithm


class RandomArm(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        
    label = "Random arm"
    
    def run(self, T, model):
        self.best_action = np.random.randint(0, model.K)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]