from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import xarray as xr

from fnorollout.data.datasets import TrajectoryDataset


def load_dataset_from_file(data_config: DictConfig) -> Dataset:
    path = data_config.path
    absolute_path = to_absolute_path(path)
    dataset = TrajectoryDataset(path=absolute_path, rollout_steps=data_config.trajectory.rollout_steps)

    return dataset
