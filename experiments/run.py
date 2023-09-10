import click

from experiments import (run_experiment_1, run_experiment_2, run_experiment_3,
                         run_experiment_4, run_experiment_5, run_experiment_6)

@click.command()
def run_experiment() -> None:
    pass

# This is the running mode code
if __name__ == "__main__":
    pass