"""Dataset classes for loading trajectory data (e.g. GeophysicalFlows.jl NetCDF output)."""

from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import Dataset


class NeuralopsTrajectoryDataset(Dataset):
    """
    Multi-trajectory dataset to track one or more fluid flow variables (e.g. vorticity, streamfunction) over time.
    Follows the dataset format required by the neuralop package trainer
    """

    def __init__(
        self,
        path: str,
        channel_names: list[str],
        rollout_steps: int = 1,
        channel_dim=1,
    ):
        """
        path: Path to directory with trajectory NetCDF files
        channel_names: Channel names per snapshot of the trajectory
        rollout_steps: Trajectory slicing
        """

        super().__init__()
        # Julia datagen scripts write NetCDF with dim order (x, y, t);
        # xarray/netCDF4 reads that reversed as (t, y, x)
        self.path = Path(path).absolute()
        self.channel_names = channel_names
        self.channel_dim = channel_dim
        self.rollout_steps = rollout_steps
        self.trajectories = self._load_trajectories()

        self.indices = [
            (traj_index, time_index)
            for traj_index, trajectory in enumerate(self.trajectories)
            for time_index in range(
                trajectory.size(0) - self.rollout_steps
            )  # 0th dimension for time
        ]  # Creates a tuple of (trajectory index, time index) for each trajectory

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        """
        Returns a snapshot with key value "x"
        and the next snapshot in the rollout step with key value "y"

        x and y are used to match with expected keys by neuralop package,
        not to be confused with x and y axes
        """
        traj_index, time_index = self.indices[index]
        trajectory = self.trajectories[traj_index]

        return {
            "x": trajectory[time_index],
            "y": trajectory[time_index + self.rollout_steps],
        }

    def _load_trajectories(self) -> list[torch.Tensor]:
        # Recursively get all NetCDF files
        file_paths = (
            [filename for filename in self.path.rglob("*.nc")]
            if self.path.is_dir()
            else [self.path]
        )

        trajectories = []
        for file_path in file_paths:
            dataset_nc = xr.open_dataset(file_path)

            # Read channel data from xarray data
            # By default only one channel
            channel_data = [
                torch.from_numpy(dataset_nc[channel_name].values).float()
                for channel_name in self.channel_names
            ]

            # Create new dim for channel
            trajectory = torch.stack(
                channel_data, dim=self.channel_dim
            )  # T x C x Y x X
            trajectories.append(trajectory)

        return trajectories
