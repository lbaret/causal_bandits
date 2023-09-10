from pathlib import Path
import yaml

from .experiment_config import ExperimentConfig


class ExperimentConfigReader(object):
    ACCEPTED_ARGUMENT_KEYS = [
        'N', 'epsilon', 'simulations', 'T', 'a', 'm',
        'pz', 'q', 'N0', 'N1', 'N2'
    ]

    @staticmethod
    def read_yaml(config_path: Path) -> ExperimentConfig:
        with open(config_path, mode='r') as f:
            config = yaml.load(f)

        return ExperimentConfig(**config)