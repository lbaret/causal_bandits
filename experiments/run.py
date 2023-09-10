from pathlib import Path

import click

from .config.experiment_config_reader import ExperimentConfigReader
from .experiment_runner import ExperimentRunner


@click.command()
@click.option("-e", "--experiment", type=int, required=True, help="Run the given experiment number (between 1 and 6)")
@click.option("-c", "--config-file", type=str, required=True, help="Path to the YAML config file")
@click.option("--verbose", type=bool, is_flag=True, default=False, help="Verbose mode")
def run_experiment(experiment: int, config_file: str, verbose: bool) -> None:
    assert experiment <= 6 and experiment >= 1
    config_path = Path(config_file)
    experiment_config = ExperimentConfigReader.read_yaml(config_path)
    
    runner = ExperimentRunner(experiment_number=experiment, experiment_config=experiment_config, verbose=verbose)
    runner.run()


if __name__ == "__main__":
    run_experiment()