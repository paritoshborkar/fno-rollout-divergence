"""
Perform rollout on a trained FNO2D model that predicts vorticity DELTAS (the
change from one snapshot to the next) rather than raw values directly. Hydra
composes configs/rollout_config.yaml - same config shape as
fnorollout.scripts.rollout_fno2d. Loads a previous training run (one trained with
data.type: delta, see fnorollout.data.datasets.TrajectoryDeltaDataset) and saves
the predicted rollout as an experiment outcome.

Usage:
    uv run python -m fnorollout.scripts.rollout_fno2d_delta artifacts.path=<train_artifacts_path> output_path=<predicted rollout path>
    uv run python -m fnorollout.scripts.rollout_fno2d_delta +rollout_experiments=experiment_1
"""

import hydra
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from omegaconf import DictConfig

from fnorollout.rollout_utils import run_rollout_and_save
from fnorollout.utils import register_utcnow_resolver

register_utcnow_resolver()


def rollout_delta(
    model: torch.nn.Module,
    data_processor: DefaultDataProcessor,
    initial_snapshots: torch.Tensor,
    rollout_steps: int = 100,
    n_input_snapshots: int = 1,
):
    """
    Run an autoregressive rollout using a trained FNO2D model that predicts deltas:
    each model.forward() call predicts a sequence of consecutive snapshot-to-
    snapshot deltas, which are cumulatively added onto the last known/reconstructed
    snapshot to recover actual predicted states.

    data_processor: Pre/post processing applied on raw trajectory snapshots
    initial_snapshots: Initial trajectory snapshots to rollout from: Shape -> 1 x C * n_input_snapshots x H x W
    rollout_steps: Number of autoregressive model.forward() calls to run
    n_input_snapshots: Number of time snapshots stacked into the channel dimension
    """

    model.eval()

    _, C, H, W = initial_snapshots.shape
    initial_snapshots = initial_snapshots.reshape(
        shape=(1, n_input_snapshots * C, H, W)
    )

    # In and Out normalizers have different statistics, 
    # so deltas must be inversed transformed first before
    # adding to raw values
    real_snapshots = list(torch.split(initial_snapshots, split_size_or_sections=C, dim=1))

    current_window = data_processor.in_normalizer.transform(initial_snapshots)

    for _ in range(rollout_steps):
        predicted_deltas = model.forward(current_window)  # 1 x rollout_steps * C x H x W, out_normalizer space
        predicted_deltas = torch.stack(
            torch.split(predicted_deltas, split_size_or_sections=C, dim=1), dim=0
        )
        predicted_deltas = data_processor.out_normalizer.inverse_transform(predicted_deltas)

        # Deltas are added to the last real valued snapshot
        predicted_snapshots = real_snapshots[-1] + predicted_deltas.cumsum(dim=0)
        real_snapshots.extend(predicted_snapshots)

        current_window = data_processor.in_normalizer.transform(
            torch.cat(real_snapshots[-n_input_snapshots:], dim=1)
        )

    # real_snapshots was never in a normalized space - no final inverse_transform needed
    predicted_trajectory = torch.cat(real_snapshots, dim=0)

    return predicted_trajectory


@hydra.main(
    version_base=None, config_path="../../configs", config_name="rollout2d_config"
)
def main(config: DictConfig):
    """
    Performs a delta-based rollout on a trained FNO2D model
    """
    run_rollout_and_save(config, rollout_fn=rollout_delta)


if __name__ == "__main__":
    main()
