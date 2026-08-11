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
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from fnorollout.scripts.data import load_dataset_from_file
from neuralop.models import FNO


def create_dataloaders(data_config: DictConfig):
    dataset = load_dataset_from_file(data_config=data_config)

    train_split = data_config.split.train
    val_split = data_config.split.val

    train_index = int(len(dataset) * train_split)
    val_index = train_index + int(len(dataset) * val_split)

    train_loader = DataLoader(
        dataset=Subset(dataset, range(train_index)),
        batch_size=data_config.dataloader.train.batch_size,
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, range(train_index, val_index)),
        batch_size=data_config.dataloader.val.batch_size,
    )

    test_loader = DataLoader(
        dataset=Subset(dataset, range(val_index, len(dataset))),
        batch_size=data_config.dataloader.test.batch_size,
    )

    test_resolution = dataset[0]["x"].shape[-1]
    test_loaders = {test_resolution: test_loader}
    return train_loader, val_loader, test_loaders


def load_optimizer(train_config: DictConfig, model):
    return instantiate(train_config.optimizer, params=model.parameters())


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    print("Loaded data, model and training configs")

    data_config = config.data
    model_config = config.model
    train_config = config.training

    train_dataloader, val_dataloader, test_dataloaders = create_dataloaders(
        data_config=data_config
    )

    print("Instantiating model")
    model: FNO = instantiate(model_config)
    model.to(device=DEVICE)
    print(f"Instantiated model {model_config._target_}")

    print("Creating optimizer")
    optimizer = load_optimizer(train_config=train_config, model=model)
    print(f"Finished loading {train_config.optimizer._target_} optimizer")


if __name__ == "__main__":
    # Create training loop
    ## Use training config values to run training loop
    ## Track experiment and save artifacts

    # Measure model performance
    ## Apart from loss define metrics to measure model performance

    # Easily configurable
    ## Running new experiments should be a matter of just changing the config files
    main()
