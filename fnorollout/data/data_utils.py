import re
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from fnorollout.constants import DataSourceType
from fnorollout.data.datasets import NeuralopsTrajectoryDataset


def load_dataset_from_path(
    data_source: DictConfig, rollout_steps: int, n_input_snapshots: int
) -> Dataset:
    """
    Loads a dataset from a local file
    """
    absolute_path = Path(to_absolute_path(data_source.path))

    file_paths = (
        [filename for filename in absolute_path.rglob("*.nc")]
        if absolute_path.is_dir()
        else [absolute_path]
    )

    if data_source.get("filter"):
        file_paths = [
            filtered_file
            for filtered_file in file_paths
            if re.search(data_source.filter, filtered_file.name)
        ]

    if not file_paths:
        raise IndexError(
            f"No valid data files present in data source {data_source.name}. Check source and filters"
        )

    return NeuralopsTrajectoryDataset(
        file_paths=file_paths,
        channel_names=list(data_source.channels),
        rollout_steps=rollout_steps,
        n_input_snapshots=n_input_snapshots,
    )


def load_dataset_from_source(
    data_source: DictConfig, rollout_steps: int, n_input_snapshots: int
) -> Dataset:
    """
    Loads a single data source's dataset according to its source type
    """
    if data_source.type == DataSourceType.LOCAL:
        return load_dataset_from_path(
            data_source=data_source,
            rollout_steps=rollout_steps,
            n_input_snapshots=n_input_snapshots,
        )

    raise NotImplementedError(
        f"Data source type '{data_source.type}' is not yet supported"
    )


def load_dataset(sources: DictConfig, rollout_steps: int) -> Dataset:
    """
    Loads and concatenates datasets from a mapping of named data sources
    """
    datasets = [
        load_dataset_from_source(data_source, rollout_steps)
        for data_source in sources.values()
    ]
    return ConcatDataset(datasets)


def split_dataset_indices(
    dataset: NeuralopsTrajectoryDataset, train_split: float, val_split: float
) -> tuple[list[int], list[int]]:
    """
    Splits a single source's flat (trajectory, timestep) items into train/val indices.

    When the dataset holds more than one trajectory, splits by whole trajectory so
    no trajectory straddles both splits. Falls back to a time-based split within the
    single trajectory when there's only one (e.g. a single-file data source).
    """
    num_trajectories = len(dataset.trajectories)

    if num_trajectories > 1:
        # round (not truncate) so e.g. 3 trajectories * 0.8 -> 2 rather than dropping
        # to a too-small train split, and always reserve at least one trajectory for
        # train even if train_split rounds down to 0 trajectories
        train_traj_count = max(
            1, min(round(num_trajectories * train_split), num_trajectories)
        )
        val_traj_count = min(
            round(num_trajectories * val_split), num_trajectories - train_traj_count
        )
        train_trajs = range(train_traj_count)
        val_trajs = range(train_traj_count, train_traj_count + val_traj_count)

        train_indices = [
            i
            for i, (traj_index, _) in enumerate(dataset.indices)
            if traj_index in train_trajs
        ]
        val_indices = [
            i
            for i, (traj_index, _) in enumerate(dataset.indices)
            if traj_index in val_trajs
        ]
    else:
        train_index = max(1, min(round(len(dataset) * train_split), len(dataset)))
        val_count = min(round(len(dataset) * val_split), len(dataset) - train_index)
        train_indices = list(range(train_index))
        val_indices = list(range(train_index, train_index + val_count))

    return train_indices, val_indices


def create_neuralop_dataloaders(
    data_config: DictConfig, train_config: DictConfig
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """
    Creates train and validation dataloaders, as required by neuralop's Trainer

    val_loaders is keyed by grid resolution (shape[-1] of a sample's "x"), matching
    the shape neuralop's Trainer expects for its `test_loaders` argument so it can
    report per-resolution eval metrics.
    """

    if data_config.get("test_sources"):
        train_datasets: list[Dataset] = [
            load_dataset_from_source(
                data_source=data_source,
                rollout_steps=data_config.trajectory.rollout_steps,
                n_input_snapshots=data_config.trajectory.n_input_snapshots,
            )
            for data_source in data_config.sources.values()
        ]

        val_datasets: list[Dataset] = [
            load_dataset_from_source(
                data_source=data_source,
                rollout_steps=data_config.trajectory.rollout_steps,
                n_input_snapshots=data_config.trajectory.n_input_snapshots,
            )
            for data_source in data_config.test_sources.values()
        ]
        val_resolutions = [dataset[0]["x"].shape[-1] for dataset in val_datasets]

        train_loader = DataLoader(
            dataset=ConcatDataset(train_datasets),
            batch_size=train_config.dataloader.train.batch_size,
        )
        val_loaders = {
            val_resolutions[index]: DataLoader(
                dataset=val_dataset, batch_size=train_config.dataloader.val.batch_size
            )
            for index, val_dataset in enumerate(val_datasets)
        }

    else:
        train_split = data_config.split.train
        val_split = data_config.split.val

        train_datasets: list[Dataset] = []
        val_datasets: list[tuple[int, Dataset]] = []
        for data_source in data_config.sources.values():
            dataset = load_dataset_from_source(
                data_source,
                data_config.trajectory.rollout_steps,
                data_config.trajectory.n_input_snapshots,
            )
            train_indices, val_indices = split_dataset_indices(
                dataset, train_split, val_split
            )
            train_datasets.append(Subset(dataset, train_indices))

            val_resolution = dataset[0]["x"].shape[-1]
            val_datasets.append((val_resolution, Subset(dataset, val_indices)))

        train_loader = DataLoader(
            dataset=ConcatDataset(train_datasets),
            batch_size=train_config.dataloader.train.batch_size,
        )
        val_loaders = (
            {
                resolution: DataLoader(
                    dataset=val_dataset,
                    batch_size=train_config.dataloader.val.batch_size,
                )
                for resolution, val_dataset in val_datasets
            }
            if val_split > 0.0
            else {}
        )

    return train_loader, val_loaders
