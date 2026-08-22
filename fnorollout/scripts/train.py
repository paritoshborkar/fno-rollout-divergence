"""Training entrypoint. Hydra composes configs/config.yaml; wandb tracks the run;
Hydra's own run dir (see hydra.run.dir in configs/config.yaml) holds per-run logs
and is where training artifacts (checkpoints) should be written.

Usage:
    uv run python -m fno_rollout_divergence.train
    uv run python -m fno_rollout_divergence.train +experiment=baseline
"""

from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from neuralop.models import FNO
from neuralop.training import Trainer
from omegaconf import DictConfig, OmegaConf

from fnorollout.schemas.configs import Config
from fnorollout.scripts.data import create_dataloaders, create_neuralop_test_dataloaders


def load_optimizer(train_config: DictConfig, model):
    return instantiate(train_config.optimizer, params=model.parameters())


def load_scheduler(train_config: DictConfig, optimizer):
    return instantiate(train_config.scheduler, optimizer=optimizer)


def init_wandb(config: DictConfig) -> None:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=list(config.wandb.tags),
        config=OmegaConf.to_container(config, resolve=True),
    )


def train_loop(
    trainer: Trainer,
    train_config: DictConfig,
    loss_config: DictConfig,
    train_dataloader,
    test_dataloaders,
    optimizer,
    scheduler,
):
    training_loss = instantiate(next(iter(loss_config.training_loss.values())))
    eval_losses = instantiate(loss_config.eval_losses)

    save_best = None
    if train_config.checkpoints.keep_best:
        loader_name = next(iter(test_dataloaders))
        loss_name = next(iter(eval_losses))
        save_best = f"{loader_name}_{loss_name}"

    return trainer.train(
        train_loader=train_dataloader,
        test_loaders=test_dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_every=train_config.checkpoints.save_every,
        save_best=save_best,
        save_dir=train_config.checkpoints.path,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    raw_config = OmegaConf.to_container(config)
    _ = Config(**raw_config) # Validates main config with pydatic

    print("Loaded data, model and training configs")

    init_wandb(config)

    data_config = config.data
    model_config = config.model
    train_config = config.training
    loss_config = config.loss

    train_dataloader, val_dataloader = create_dataloaders(
        data_config=data_config, train_config=train_config
    )
    test_dataloaders = create_neuralop_test_dataloaders(
        data_config=data_config, train_config=train_config
    )

    print("Instantiating model")
    model: FNO = instantiate(model_config)
    model.to(device=DEVICE)
    print(f"Instantiated model {model_config._target_}")

    print("Creating optimizer and scheduler")
    optimizer = load_optimizer(train_config=train_config, model=model)
    scheduler = load_scheduler(train_config=train_config, optimizer=optimizer)
    print(f"Finished loading {train_config.optimizer._target_} optimizer")
    print(f"Finished loading {train_config.scheduler._target_} scheduler")

    print("Creating trainer")
    trainer = Trainer(
        model=model,
        n_epochs=train_config.epochs,
        device=DEVICE,
        wandb_log=True,
        verbose=True,
    )

    print("Starting training loop")
    train_loop(
        trainer=trainer,
        train_config=train_config,
        loss_config=loss_config,
        train_dataloader=train_dataloader,
        test_dataloaders=test_dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    wandb.finish()


if __name__ == "__main__":
    # Measure model performance
    ## Apart from loss define metrics to measure model performance

    # Easily configurable
    ## Running new experiments should be a matter of just changing the config files
    main()
