"""Dataset classes for loading trajectory data (e.g. GeophysicalFlows.jl NetCDF output)."""

from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Dataset to track a fluid flow variable like vorticity over time.
    Follows the dataset format required by the neuralop package trainer
    """

    def __init__(self, path: Path, rollout_steps: int = 1, channel_dim=1):
        super().__init__()
        dataset_nc = xr.open_dataset(path)  # y, x, time index order

        zeta = torch.from_numpy(dataset_nc["zeta"].values).float()  # Make float32 dtype
        zeta = zeta.permute(2, 1, 0)  # y,x,time -> time,x,y
        self.zeta = zeta.unsqueeze(channel_dim)  # time,x,y -> time, channel=1, x, y
        self.rollout_steps = rollout_steps

    def __len__(self):
        return self.zeta.shape[0] - self.rollout_steps

    def __getitem__(self, index: int):
        """
        Returns a snapshot with key value "x"
        and the next snapshot in the rollout step with key value "y"

        x and y are used to match with expected keys by neuralop package,
        not to be confused with x and y axes
        """
        return {
            "x": self.zeta[index], 
            "y": self.zeta[index + self.rollout_steps]
            }
