from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from fnorollout.constants import DataSourceType
from fnorollout.data.datasets import TrajectoryDataset


def load_dataset_from_file(data_source: DictConfig, rollout_steps: int) -> Dataset:
    """
    Loads a dataset from a local file
    """
    absolute_path = to_absolute_path(data_source.path)
    return TrajectoryDataset(
        path=absolute_path,
        channel_names=list(data_source.channels),
        rollout_steps=rollout_steps,
    )


def load_dataset_from_source(data_source: DictConfig, rollout_steps: int) -> Dataset:
    """
    Loads a single data source's dataset according to its source type
    """
    if data_source.type == DataSourceType.LOCAL:
        return load_dataset_from_file(
            data_source=data_source, rollout_steps=rollout_steps
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


def create_dataloaders(data_config: DictConfig, train_config: DictConfig):
    """
    Creates train and validation dataloaders
    """
    dataset = load_dataset(data_config.sources, data_config.trajectory.rollout_steps)

    train_split = data_config.split.train
    val_split = data_config.split.val

    train_index = int(len(dataset) * train_split)
    val_index = train_index + int(len(dataset) * val_split)

    train_loader = DataLoader(
        dataset=Subset(dataset, range(train_index)),
        batch_size=train_config.dataloader.train.batch_size,
    )
    val_loader = DataLoader(
        dataset=Subset(dataset, range(train_index, val_index)),
        batch_size=train_config.dataloader.val.batch_size,
    )

    return train_loader, val_loader


def create_neuralop_test_dataloaders(
    data_config: DictConfig, train_config: DictConfig
) -> dict:
    """
    Creates test dataloaders keyed by resolution, matching the shape neuralop's Trainer
    expects for its `test_loaders` argument during training
    """
    test_dataset = load_dataset(
        data_config.test_sources, data_config.trajectory.rollout_steps
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_config.dataloader.test.batch_size,
    )

    test_resolution = test_dataset[0]["x"].shape[-1]
    return {test_resolution: test_loader}
