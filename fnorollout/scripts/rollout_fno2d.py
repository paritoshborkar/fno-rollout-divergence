"""
Perform rollout on a trained FNO2D model that predicts raw snapshot values
directly. Hydra composes configs/rollout_config.yaml. Loads a previous training
run and saves the predicted rollout as an experiment outcome.

For a model trained on vorticity deltas instead (data.type: delta, see
fnorollout.data.datasets.TrajectoryDeltaDataset), use
fnorollout.scripts.rollout_fno2d_delta instead.

Usage:
    uv run python -m fnorollout.scripts.rollout_fno2d artifacts.path=<train_artifacts_path> output_path=<predicted rollout path>
    uv run python -m fnorollout.scripts.rollout_fno2d +rollout_experiments=experiment_1
"""

import hydra
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from omegaconf import DictConfig

from fnorollout.rollout_utils import run_rollout_and_save
from fnorollout.utils import register_utcnow_resolver

register_utcnow_resolver()


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
    rollout_steps: Number of autoregressive model.forward() calls to run
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
        predicted_snapshots = model.forward(
            current_window
        )  # 1 x rollout_steps * C x H x W
        predicted_snapshots = list(
            torch.split(predicted_snapshots, split_size_or_sections=C, dim=1)
        )
        snapshots.extend(predicted_snapshots)

        current_window = torch.cat(snapshots[-n_input_snapshots:], dim=1)

    # Apply data postrocessing
    predicted_trajectory = torch.cat(snapshots, dim=0)
    predicted_trajectory = data_processor.out_normalizer.inverse_transform(
        predicted_trajectory
    )

    return predicted_trajectory


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rollout2d_config"
)
def main(config: DictConfig):
    """
    Performs a rollout on a trained FNO2D model
    """
    run_rollout_and_save(config, rollout_fn=rollout)


if __name__ == "__main__":
    main()
