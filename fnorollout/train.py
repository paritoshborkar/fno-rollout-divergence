"""Training entrypoint. Hydra composes configs/config.yaml; wandb tracks the run;
Hydra's own run dir (see hydra.run.dir in configs/config.yaml) holds per-run logs
and is where training artifacts (checkpoints) should be written.

Usage:
    uv run python -m fno_rollout_divergence.train
    uv run python -m fno_rollout_divergence.train +experiment=baseline
"""

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    print("Loaded data, model and training configs")

    data_config = config.data
    model_config = config.model
    train_config = config.training


if __name__ == "__main__":
    # Load data
    ## Using data config values, load the correct data file
    ## Convert from NETcdf to pytorch tensors

    # Preprocessing
    ## Normalize data, convert data from column to row major
    ## Add layer of positional coordinates

    # Instantiate model and optimizer

    # Create training loop
    ## Use training config values to run training loop
    ## Track experiment and save artifacts

    # Measure model performance
    ## Apart from loss define metrics to measure model performance

    # Easily configurable
    ## Running new experiments should be a matter of just changing the config files
    main()
