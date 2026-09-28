import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fnorollout.stats.trajectory_statistics import Vorticity2DTrajectoryStatistics


def load_netcdf_to_torch(
    netcdf_filepath: Path, channel: str | None = None, channel_dim: int = 1
) -> tuple[torch.Tensor, float, float, float]:
    """
    Reads NetCDF file and returns data as a torch Tensor
    """
    dataset_nc = xr.open_dataset(netcdf_filepath.absolute())

    x_axis = dataset_nc["x"].to_numpy()
    y_axis = dataset_nc["y"].to_numpy()

    # Assumes only one channel in the file
    if not channel:
        channel = next(iter(dataset_nc.data_vars))

    return (
        torch.from_numpy(dataset_nc[channel].values).float().unsqueeze(channel_dim),
        x_axis,
        y_axis,
    )


def get_trajectory_statistics(
    trajectory: Path, x_axis: np.ndarray, y_axis: np.ndarray
) -> Vorticity2DTrajectoryStatistics:
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]

    return Vorticity2DTrajectoryStatistics(trajectory=trajectory, dx=dx, dy=dy)


def get_comparative_scalar_metrics(
    real_stats: Vorticity2DTrajectoryStatistics,
    pred_stats: Vorticity2DTrajectoryStatistics,
) -> str:
    time_steps = real_stats.trajectory.shape[0]
    rms_vorticity = real_stats.vorticity_rms()

    early_time_window = (0, int(time_steps * 0.2))
    mid_time_window = (int(time_steps * 0.2), int(time_steps * 0.7))
    end_time_window = (int(time_steps * 0.7), time_steps)
    ref_wasserstein_dist = (
        real_stats.wasserstein_1d(
            real_stats.trajectory,
            pred_window=end_time_window,
            baseline_window=early_time_window,
        )
        / rms_vorticity
    )
    wasserstein_distance_start = (
        pred_stats.wasserstein_1d(
            real_stats.trajectory,
            pred_window=early_time_window,
            baseline_window=early_time_window,
        )
        / rms_vorticity
    )
    wasserstein_distance_mid = (
        pred_stats.wasserstein_1d(
            real_stats.trajectory,
            pred_window=mid_time_window,
            baseline_window=mid_time_window,
        )
        / rms_vorticity
    )
    wasserstein_distance_end = (
        pred_stats.wasserstein_1d(
            real_stats.trajectory,
            pred_window=end_time_window,
            baseline_window=end_time_window,
        )
        / rms_vorticity
    )

    return (
        "Wasserstein distance / RMS vorticity:\n"
        f"  reference (real vs itself, t={end_time_window} vs t={early_time_window}: {ref_wasserstein_dist.item():.4f}\n"
        f"  predicted vs real, t={early_time_window}:   {wasserstein_distance_start.item():.4f}\n"
        f"  predicted vs real, t={mid_time_window}: {wasserstein_distance_mid.item():.4f}\n"
        f"  predicted vs real, t={end_time_window}: {wasserstein_distance_end.item():.4f}"
    )


def plot_comparisons(
    pred_stats: Vorticity2DTrajectoryStatistics,
    real_stats: Vorticity2DTrajectoryStatistics,
):
    # Predicted trajcetory statistics
    pred_enstrophy_per_snapshot = pred_stats.enstrophy_per_snapshot()
    pred_energy_density_per_snapshot = pred_stats.energy_density_per_snapshot()

    # Real trajectory statistics
    real_enstrophy_per_snapshot = real_stats.enstrophy_per_snapshot()
    real_energy_density_per_snapshot = real_stats.energy_density_per_snapshot()

    # Comparative statistics
    time_steps = real_stats.trajectory.shape[0]
    energy_drift_per_snapshot = pred_stats.energy_drift_per_snapshot(
        real_stats.trajectory
    )

    metrics_summary = get_comparative_scalar_metrics(
        real_stats=real_stats, pred_stats=pred_stats
    )
    print(metrics_summary)

    t_axis = np.arange(0, time_steps)

    fig, axes = plt.subplots(1, 3, subplot_kw={"box_aspect": 1}, figsize=(11, 5))

    # Plot Enstrophy per snapshot: predicted vs real
    axes[0].plot(t_axis, pred_enstrophy_per_snapshot, color="b", label="predicted")
    axes[0].plot(t_axis, real_enstrophy_per_snapshot, color="r", label="real")
    axes[0].set_title("Enstrophy per snapshot")
    axes[0].set_xlabel("timestep")
    axes[0].legend()

    # Plot Energy Density per snapshot: predicted vs real
    axes[1].plot(t_axis, pred_energy_density_per_snapshot, color="b", label="predicted")
    axes[1].plot(t_axis, real_energy_density_per_snapshot, color="r", label="real")
    axes[1].set_title("Energy Density per snapshot")
    axes[1].set_xlabel("timestep")
    axes[1].legend()

    axes[2].plot(t_axis, energy_drift_per_snapshot, color="r")
    axes[2].set_title("Energy drift per snapshot")
    axes[2].set_xlabel("timestep")

    # Attach metrics summary to figure
    fig.text(0.02, 0.02, metrics_summary, fontsize=8, family="monospace", va="bottom")
    fig.subplots_adjust(bottom=0.32)

    plt.show()


def main(cmd_args):
    predicted_filepath = Path(cmd_args.pred)
    real_filepath = Path(cmd_args.real)
    channel = cmd_args.channel
    channel_dim = int(cmd_args.cdim)

    pred_trajectory, x_axis, y_axis = load_netcdf_to_torch(
        netcdf_filepath=predicted_filepath, channel=channel, channel_dim=channel_dim
    )
    real_trajectory, x_axis, y_axis = load_netcdf_to_torch(
        netcdf_filepath=real_filepath, channel=channel, channel_dim=channel_dim
    )

    pred_stats = get_trajectory_statistics(
        trajectory=pred_trajectory, x_axis=x_axis, y_axis=y_axis
    )
    real_stats = get_trajectory_statistics(
        trajectory=real_trajectory,
        x_axis=x_axis,
        y_axis=y_axis,
    )

    plot_comparisons(pred_stats=pred_stats, real_stats=real_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Path to NetCDF file of predicted trajectory")
    parser.add_argument(
        "--real", help="Path to NetCDF file of real/simulated trajectory"
    )
    parser.add_argument(
        "--channel",
        default="zeta",
        help="Channel name for vorticity data in the NetCDF files",
    )
    parser.add_argument(
        "--cdim",
        default=1,
        help="Channel dimension for vorticity data in the NetCDF files",
    )

    args = parser.parse_args()
    main(cmd_args=args)
