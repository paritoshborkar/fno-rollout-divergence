"""Dataset classes for loading trajectory data (e.g. GeophysicalFlows.jl NetCDF output)."""

from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Dataset to track one or more fluid flow variables (e.g. vorticity, streamfunction) over time.
    Follows the dataset format required by the neuralop package trainer
    """

    def __init__(self, path: Path, channel_names: list[str], rollout_steps: int = 1, channel_dim=1):
        super().__init__()
        dataset_nc = xr.open_dataset(path)  # y, x, time index order

        channels = [
            torch.from_numpy(dataset_nc[channel_name].values).float().permute(2, 1, 0)  # y,x,time -> time,x,y
            for channel_name in channel_names
        ]
        self.trajectory = torch.stack(channels, dim=channel_dim)  # time,x,y -> time, channel, x, y
        self.rollout_steps = rollout_steps

    def __len__(self):
        return self.trajectory.shape[0] - self.rollout_steps

    def __getitem__(self, index: int):
        """
        Returns a snapshot with key value "x"
        and the next snapshot in the rollout step with key value "y"

        x and y are used to match with expected keys by neuralop package,
        not to be confused with x and y axes
        """
        return {
            "x": self.trajectory[index],
            "y": self.trajectory[index + self.rollout_steps]
            }
