"""Preprocessing/normalization transforms applied to trajectory data before training."""

from pathlib import Path

import torch
from hydra.utils import instantiate
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from omegaconf import DictConfig
from torch.utils.data import Dataset


def build_data_processor(
    train_dataset: Dataset, preprocessing_config: DictConfig
) -> DefaultDataProcessor:
    """
    Fits normalizers (via `preprocessing_config.normalizer`'s `_target_`, e.g.
    UnitGaussianNormalizer.from_dataset) on a TrajectoryDataset's "x" and "y" fields
    (each a channel, height, width snapshot) and wraps them in a DataProcessor for use
    with neuralop's Trainer, which normalizes inputs/targets before the forward pass
    and un-normalizes predictions before eval losses are computed.

    `_target_` is expected to be a `from_dataset(dataset, keys, ...)`-style factory
    returning a dict keyed by field name — agnostic to which normalizer type
    configs/data/default.yaml's `preprocessing.normalizer` selects, as long as it fits
    incrementally (e.g. via `partial_fit`) rather than materializing the whole dataset.
    """
    normalizers = instantiate(
        preprocessing_config.normalizer,
        dataset=train_dataset,
        keys=list(preprocessing_config.fields),
    )
    return DefaultDataProcessor(
        in_normalizer=normalizers["x"], out_normalizer=normalizers["y"]
    )


def save_data_processor(data_processor: DefaultDataProcessor, save_dir: str | Path) -> Path:
    """
    Save a fitted DataProcessor's normalizer stats (mean/std buffers) so it can be
    reloaded later to preprocess a different test set the same way training data was.
    """
    save_path = Path(save_dir) / "data_processor.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_processor.state_dict(), save_path)
    return save_path
