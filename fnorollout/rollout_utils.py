"""
Shared utilities for FNO2D rollout entrypoints
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from hydra.utils import get_class, instantiate, to_absolute_path
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from omegaconf import DictConfig, OmegaConf

from fnorollout.schemas.data_source import LocalDataSource
from fnorollout.schemas.rollout_config import Rollout2DConfig
from fnorollout.stats.trajectory_metrics import Vorticity2DTrajectoryMetrics


def load_train_run_config(artifacts: DictConfig) -> DictConfig:
    """
    Loading merged config file used for training the model
    """
    train_run_config_path = (
        Path(to_absolute_path(artifacts.path)) / artifacts.train_config
    )
    return OmegaConf.load(train_run_config_path)


def load_data_processor(
    data_config: DictConfig, artifacts: DictConfig
) -> DefaultDataProcessor:
    """
    Loads a neuralops DefaultDataProcessor that was created as part of a previous
    training run
    """
    artifacts_path = Path(to_absolute_path(artifacts.path))
    state_dict = torch.load(artifacts_path / artifacts.data_processor)

    normalizer_config = data_config.preprocessing.normalizer
    normalizer_cls = get_class(normalizer_config._target_.rsplit(".", 1)[0])

    def build_normalizer(prefix: str):
        return normalizer_cls(
            mean=state_dict[f"{prefix}.mean"],
            std=state_dict[f"{prefix}.std"],
            dim=list(normalizer_config.dim),
        )

    return DefaultDataProcessor(
        in_normalizer=build_normalizer("in_normalizer"),
        out_normalizer=build_normalizer("out_normalizer"),
    )


def load_test_data(
    test_data_source: LocalDataSource, channel_dim=1
) -> tuple[torch.Tensor, xr.Dataset]:
    """
    test_filepath: Path to NetCDF file with trajectory data
    Load test trajectory data, returning both the tensor (for the model) and the
    source xarray Dataset (its t/y/x coordinate values are reused to build
    coordinates for the saved predicted trajectory).
    """

    test_filepath = Path(to_absolute_path(str(test_data_source.path)))
    dataset_nc = xr.open_dataset(test_filepath)

    channel_data = [
        torch.from_numpy(dataset_nc[channel_name].values).float()
        for channel_name in test_data_source.channels
    ]

    # Create new dim for channel
    trajectory = torch.stack(channel_data, dim=channel_dim)  # T x C x Y x X
    return trajectory, dataset_nc


def get_initial_snapshots(
    trajectory: torch.Tensor, start_index: int = 0, n_input_snapshots: int = 1
) -> torch.Tensor:
    """
    Returns the first snapshot from a trajectory

    returns shape -> T x C x H x W
    """
    return trajectory[start_index - n_input_snapshots + 1 : start_index + 1, ...]


def instantiate_model(
    model_config: DictConfig, artifacts: DictConfig, device: str
) -> torch.nn.Module:
    """
    Instantiate model from trained model dict
    """
    model = instantiate(model_config)
    artifacts_path = Path(to_absolute_path(artifacts.path))

    model_filepath = artifacts_path / artifacts.model
    model.load_state_dict(
        torch.load(model_filepath, map_location=device, weights_only=False)
    )

    return model


def compute_rollout_iterations(
    total_rollout_snapshots: int, model_rollout_steps: int
) -> int:
    """
    Calculates number of iterations required to produce desired number of snapshots
    """
    return -(-total_rollout_snapshots // model_rollout_steps)


def build_rollout_coords_from_trajectory(
    source: xr.Dataset, n_timesteps: int
) -> dict[str, np.ndarray]:
    """
    Builds t/y/x coordinate arrays for a predicted rollout, based on the source
    trajectory's time step size
    """
    source_t = source["t"].values
    t0 = source_t[0]
    dt = source_t[1] - source_t[0]
    t = t0 + dt * np.arange(n_timesteps)

    return {"t": t, "y": source["y"].values, "x": source["x"].values}


def save_predicted_trajectory(
    predicted_trajectory: torch.Tensor,
    channel_names: list[str],
    coords: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Saves a predicted rollout trajectory (T x C x Y x X) to a NetCDF file, one data
    variable per channel with dims (t, y, x)
    """
    trajectory = predicted_trajectory.detach().cpu().numpy()
    data_vars = {
        channel_name: (("t", "y", "x"), trajectory[:, channel_index, :, :])
        for channel_index, channel_name in enumerate(channel_names)
    }
    dataset = xr.Dataset(data_vars, coords=coords)

    output_path = Path(output_path).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)


def find_divergence_step(trajectory: torch.Tensor) -> int | None:
    """
    Finds the first timestep index where the trajectory stops being finite (NaN or Inf
    anywhere in that snapshot).
    None if it stays finite throughout
    """
    finite_per_step = torch.isfinite(trajectory).flatten(start_dim=1).all(dim=1)
    nonfinite_steps = (~finite_per_step).nonzero(as_tuple=True)[
        0
    ]  # List of timestep indices with NaN/inf values
    return int(nonfinite_steps[0]) if len(nonfinite_steps) > 0 else None


def compute_rollout_metrics(
    predicted_trajectory: torch.Tensor,
    real_trajectory: torch.Tensor,
    t_coords: np.ndarray,
    rollout_fn: Callable[..., torch.Tensor],
    divergence_step: int | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> dict:
    """
    Computes per-timestep and summary comparison metrics between a predicted
    rollout and its matching ground-truth trajectory (same shape, same absolute
    time range). Returns a plain-Python, JSON-serializable dict

    If the predicted trajectory diverges (NaN/Inf), metrics are computed only up
    to (not including) the first non-finite snapshot
    """
    end = divergence_step if divergence_step else predicted_trajectory.shape[0]

    pred = predicted_trajectory[:end]
    real = real_trajectory[:end]
    t = np.asarray(t_coords[:end]).tolist()

    pred_stats = Vorticity2DTrajectoryMetrics(pred, dx=dx, dy=dy)
    real_stats = Vorticity2DTrajectoryMetrics(real, dx=dx, dy=dy)

    k_bins, pred_vorticity_psd = pred_stats.psd_per_snapshot()
    _, real_vorticity_psd = real_stats.psd_per_snapshot()
    _, pred_ke_spectrum = pred_stats.kinetic_energy_psd_per_snapshot()
    _, real_ke_spectrum = real_stats.kinetic_energy_psd_per_snapshot()

    return {
        "rollout_script": f"{rollout_fn.__module__}.{rollout_fn.__qualname__}",
        "divergence_step": divergence_step,
        "metrics_over_time": {
            "t": t,
            "enstrophy": {
                "predicted": pred_stats.enstrophy_per_snapshot().tolist(),
                "real": real_stats.enstrophy_per_snapshot().tolist(),
            },
            "energy_density": {
                "predicted": pred_stats.energy_density_per_snapshot().tolist(),
                "real": real_stats.energy_density_per_snapshot().tolist(),
            },
            "relative_l2": pred_stats.relative_l2_per_snapshot(real).tolist(),
            "h1_loss": pred_stats.h1_per_snapshot(real).tolist(),
            "wasserstein_1d": [
                float(
                    pred_stats.wasserstein_1d(
                        real, pred_window=(i, i + 1), baseline_window=(i, i + 1)
                    )
                )
                for i in range(end)
            ],
        },
        "spectra": {
            "k_bins": k_bins.tolist(),
            "vorticity_psd": {
                "predicted": pred_vorticity_psd.tolist(),
                "real": real_vorticity_psd.tolist(),
            },
            "kinetic_energy_spectrum": {
                "predicted": pred_ke_spectrum.tolist(),
                "real": real_ke_spectrum.tolist(),
            },
        },
        "summary": {
            "total_enstrophy_predicted": pred_stats.total_enstrophy().item(),
            "total_enstrophy_real": real_stats.total_enstrophy().item(),
            "relative_l2_mean": pred_stats.relative_l2(real).item(),
            "energy_drift_rms": pred_stats.energy_drift(real).item(),
            "wasserstein_distance_overall": float(pred_stats.wasserstein_1d(real)),
            "vorticity_rms_predicted": pred_stats.vorticity_rms(),
            "vorticity_rms_real": real_stats.vorticity_rms(),
        },
    }


def save_rollout_metrics(metrics: dict, output_path: Path) -> Path:
    """
    Saves rollout comparison metrics (see compute_rollout_metrics) as JSON. Meant
    to be written alongside the predicted trajectory NetCDF, in the same directory.
    """
    output_path = Path(output_path).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return output_path


def run_rollout_and_save(
    config: DictConfig,
    rollout_fn: Callable[..., torch.Tensor],
) -> None:
    """
    Runs a rollout on test data and saves the predicted trajectory
    """
    device = (
        "cuda" if (torch.cuda.is_available() and config.device == "cuda") else "cpu"
    )
    print(f"Using device: {device}")

    # Load rollout configs
    raw_config = OmegaConf.to_container(config)
    _ = Rollout2DConfig(**raw_config)  # Validates main config with pydantic

    train_run_config = load_train_run_config(artifacts=config.artifacts)
    data_config = train_run_config.data
    model_config = train_run_config.model

    # Load data processor
    data_processor = load_data_processor(
        artifacts=config.artifacts, data_config=data_config
    )

    test_data_source = config.data_sources
    test_trajectory, test_dataset_nc = load_test_data(test_data_source=test_data_source)

    start_index = config.start_index
    n_input_snapshots = data_config.trajectory.n_input_snapshots

    total_rollout_snapshots = config.rollout_snapshots
    model_rollout_steps = data_config.trajectory.rollout_steps
    num_iterations = compute_rollout_iterations(
        total_rollout_snapshots, model_rollout_steps
    )

    # Get initial snapshots to rollout from
    initial_snapshots = get_initial_snapshots(
        trajectory=test_trajectory,
        start_index=start_index,
        n_input_snapshots=n_input_snapshots,
    )

    model = instantiate_model(
        model_config=model_config, artifacts=config.artifacts, device=device
    )

    print(f"Performing FNO2D rollout for {total_rollout_snapshots} time steps")
    predicted_trajectory = rollout_fn(
        model=model,
        data_processor=data_processor,
        initial_snapshots=initial_snapshots,
        rollout_steps=num_iterations,
        n_input_snapshots=n_input_snapshots,
    )
    predicted_trajectory = predicted_trajectory[
        : n_input_snapshots + total_rollout_snapshots
    ]

    # Find timestep before prediction blows up (NaN/inf) and save only
    # those snapshots that came before it
    divergence_step = find_divergence_step(predicted_trajectory)
    if divergence_step is not None:
        print(
            f"Rollout diverged to non-finite values at step {divergence_step} - "
            f"saving only snapshots predicted before it"
        )
        predicted_trajectory = predicted_trajectory[:divergence_step]

    output_path = Path(config.output_path)
    coords = build_rollout_coords_from_trajectory(
        source=test_dataset_nc, n_timesteps=predicted_trajectory.shape[0]
    )

    save_predicted_trajectory(
        predicted_trajectory=predicted_trajectory,
        channel_names=test_data_source.channels,
        coords=coords,
        output_path=output_path,
    )
    print(f"Saved predicted trajectory to {output_path.absolute()}")

    # Truncate test trajectory and match with predicted
    real_baseline = test_trajectory[
        start_index - n_input_snapshots + 1 : start_index + 1 + total_rollout_snapshots
    ]
    compare_length = min(predicted_trajectory.shape[0], real_baseline.shape[0])

    metrics = compute_rollout_metrics(
        predicted_trajectory=predicted_trajectory[:compare_length],
        real_trajectory=real_baseline[:compare_length],
        t_coords=coords["t"][:compare_length],
        rollout_fn=rollout_fn,
        divergence_step=divergence_step
    )
    metrics_path = save_rollout_metrics(
        metrics, output_path.parent / "rollout_metrics.json"
    )
    print(f"Saved rollout metrics to {metrics_path}")
