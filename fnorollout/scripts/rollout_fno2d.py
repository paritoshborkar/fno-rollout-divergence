"""
Perform rollout on a trained FNO2D model. Hydra composes configs/rollout_config.yaml
Loads a previous training run and saves the predicted rollout as an experiment outcome

Usage:
    uv run python -m fnorollout.scripts.rollout_fno2d artifacts.path=<train_artifacts_path> output_path=<predicted rollout path>
    uv run python -m fnorollout.scripts.rollout_fno2d +rollout_experiments=experiment_1
"""

from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import get_class, instantiate, to_absolute_path
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from omegaconf import DictConfig, OmegaConf

from fnorollout.schemas.data_source import LocalDataSource
from fnorollout.schemas.rollout_config import Rollout2DConfig
from fnorollout.utils import register_utcnow_resolver

register_utcnow_resolver()


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


def rollout(
    model: torch.nn.Module,
    data_processor: DefaultDataProcessor,
    initial_snapshots: torch.Tensor,
    rollout_steps: int = 100,
    n_input_snapshots: int = 1,
):
    """
    Run an autoregressive rollout using a trained FNO2D model.

    data_processor: Pre/post processing applied on raw trajectory snapshots
    initial_snapshots: Initial trajectory snapshots to rollout from: Shape -> 1 x C * n_input_snapshots x H x W
    rollout_steps: Number of time steps to rollout from the initial snapshots
    n_input_snapshots: Number of time snapshots stacked into the channel dimension
    """

    model.eval()

    # Stack initial snapshots first, then apply normalization
    _, C, H, W = initial_snapshots.shape

    initial_snapshots = initial_snapshots.reshape(
        shape=(1, n_input_snapshots * C, H, W)
    )
    normalized_snapshots = data_processor.in_normalizer.transform(initial_snapshots)

    # Individual (1, C, H, W) snapshots, oldest first - the window is at the end
    snapshots = list(torch.split(normalized_snapshots, split_size_or_sections=C, dim=1))

    current_window = normalized_snapshots

    for _ in range(rollout_steps):
        predicted_snapshots = model.forward(current_window) # 1 x rollout_steps * C x H x W
        predicted_snapshots = list(torch.split(predicted_snapshots, split_size_or_sections=C, dim=1))
        snapshots.extend(predicted_snapshots)

        current_window = torch.cat(snapshots[-n_input_snapshots:], dim=1)

    # Apply data postrocessing
    predicted_trajectory = torch.cat(snapshots, dim=0)
    predicted_trajectory = data_processor.out_normalizer.inverse_transform(
        predicted_trajectory
    )

    return predicted_trajectory


def build_rollout_coords_from_trajcetory(
    source: xr.Dataset, n_timesteps: int
) -> dict[str, np.ndarray]:
    """
    Builds t/y/x coordinate arrays for a predicted rollout, based on the source
    trajectory the initial snapshot was taken from: y/x reuse the source's spatial
    grid unchanged, and t starts at the initial snapshot's own timestamp
    (source["t"][0], since the rollout's first frame *is* the initial snapshot) and
    continues at the source's uniform step size for n_timesteps frames.
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


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rollout2d_config"
)
def main(config: DictConfig):
    """
    Performs a rollout on a trained FNO2D model
    """
    device = (
        "cuda" if (torch.cuda.is_available() and config.device == "cuda") else "cpu"
    )
    print(f"Using device: {device}")

    raw_config = OmegaConf.to_container(config)
    _ = Rollout2DConfig(**raw_config)  # Validates main config with pydatic

    train_run_config = load_train_run_config(artifacts=config.artifacts)
    data_config = train_run_config.data
    model_config = train_run_config.model

    data_processor = load_data_processor(
        artifacts=config.artifacts, data_config=data_config
    )

    test_data_source = config.data_sources
    test_trajectory, test_dataset_nc = load_test_data(test_data_source=test_data_source)

    start_index = config.start_index
    rollout_steps = config.rollout_steps
    n_input_snapshots = data_config.trajectory.n_input_snapshots

    initial_snapshots = get_initial_snapshots(
        trajectory=test_trajectory,
        start_index=start_index,
        n_input_snapshots=n_input_snapshots,
    )

    model = instantiate_model(
        model_config=model_config, artifacts=config.artifacts, device=device
    )

    print(f"Performing FNO2D rollout for {rollout_steps} time steps")
    predicted_trajectory = rollout(
        model=model,
        data_processor=data_processor,
        initial_snapshots=initial_snapshots,
        rollout_steps=rollout_steps,
        n_input_snapshots=n_input_snapshots,
    )

    output_path = Path(config.output_path)
    coords = build_rollout_coords_from_trajcetory(
        source=test_dataset_nc, n_timesteps=predicted_trajectory.shape[0]
    )

    save_predicted_trajectory(
        predicted_trajectory=predicted_trajectory,
        channel_names=test_data_source.channels,
        coords=coords,
        output_path=output_path,
    )
    print(f"Saved predicted trajectory to {output_path.absolute()}")


if __name__ == "__main__":
    main()
