# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`fno-rollout-divergence` is a research project studying rollout divergence in Fourier Neural Operators (FNOs). It uses PyTorch + `neuraloperator` for model implementation, Hydra + OmegaConf for experiment configuration, pydantic for config validation, and Weights & Biases for experiment tracking.

## Package Manager

This project uses `uv`. Do not use `pip` directly.

```bash
uv sync                  # install all dependencies
uv add <package>         # add a new dependency
uv run python <script>   # run a script in the venv
```

## Key Dependencies

| Package | Role |
|---|---|
| `torch` | Model training and inference |
| `neuraloperator` | FNO model implementations, `Trainer`, losses (`H1Loss`, `LpLoss`) |
| `hydra-core` + `omegaconf` | Hierarchical config management for experiments |
| `pydantic` | Validates the composed Hydra config (`fnorollout/schemas/`) |
| `wandb` | Experiment tracking and logging |
| `matplotlib` | Plotting/visualization |
| `xarray` + `netcdf4` | Reading NetCDF trajectory data (e.g. GeophysicalFlows.jl output) |

## Repository Structure

```
configs/                        # Hydra config groups
  config.yaml                    #   root: defaults list (data, model, training, loss, eval), hydra.run.dir, wandb settings
  data/default.yaml               #   selects named sources from data_sources/ for train/test, train/val split
  data_sources/                   #   reusable named data source defs (local file, package, url); referenced from
                                   #   configs/data/*.yaml via Hydra defaults + `@` package overrides, e.g.
                                   #   `- /data_sources@sources.<name>: <name>`
  model/fno2d.yaml
  training/default.yaml           #   optimizer, scheduler, dataloader batch sizes, training_loss, eval_losses, checkpoints
  loss/                           #   standalone h1.yaml/l2.yaml — not currently referenced by anything; training's
                                   #   own training_loss/eval_losses inline their own _target_s instead
  eval/default.yaml               #   long-horizon rollout eval settings — not yet wired into any script
  experiment/baseline.yaml        #   +experiment=<name> override bundles (use `# @package _global_`)

fnorollout/                      # Python package (flat layout, no src/), imported as `fnorollout.*`
  data/                           #   TrajectoryDataset (multi-channel NetCDF loader) + preprocessing (stub)
  models/                         #   FNO construction from the `model` config group (build_fno2d)
  schemas/                        #   pydantic models validating the composed config (Config, DataConfig, DataSource)
  scripts/
    train.py                       #   @hydra.main entrypoint: builds model/optimizer/scheduler/Trainer,
                                    #   runs training via neuralop's Trainer, logs to wandb
    data.py                        #   loads datasets from configured sources (dispatched by DataSourceType),
                                    #   builds train/val/test dataloaders
  julia/datagen/                  #   Julia toolchain (juliaup) for generating trajectory data with
                                   #   GeophysicalFlows.jl; own Project.toml/Manifest.toml, independent of
                                   #   the uv-managed Python environment

scripts/                         # Top-level, NOT part of the fnorollout package — don't confuse with fnorollout/scripts/
  plot_vorticity.py                #   quick NetCDF vorticity plotting utility
  setup_remote.sh                   #   bootstrap script for a fresh GPU instance

notebooks/                       # Exploratory Jupyter notebooks (prototyping only, not the source of truth)

data/
  raw/, processed/                # gitignored except .gitkeep; processed/torus2d_trajectory.nc is the dataset
                                   # referenced by configs/data_sources/torus2d_example.yaml's `path`
```

Data source `type` handling in `fnorollout/scripts/data.py` only implements `DataSourceType.LOCAL` — `URL` and `PACKAGE` sources (e.g. `configs/data_sources/neuralop_darcy.yaml`) raise `NotImplementedError`. Check `fnorollout/scripts/data.py` before assuming a non-local source actually loads.

## Configuration

Hydra is the config system, configs live under `configs/` as shown above. Run scripts typically accept `+experiment=<name>` or `hydra.run.dir=...` overrides on the CLI, e.g.:

```bash
uv run python -m fnorollout.scripts.train +experiment=baseline
uv run python -m fnorollout.scripts.train wandb.mode=offline training.epochs=5
```

Hydra's `hydra.run.dir` (`outputs/<date>/<time>/`, gitignored) is the per-run working directory (`hydra.job.chdir: true`) — logs and training artifacts (e.g. `training.checkpoints.path`, currently `checkpoints`) should be written there rather than to a separate top-level directory.

`fnorollout/scripts/train.py`'s `main()` validates the fully composed config against `fnorollout.schemas.configs.Config` (a pydantic model) before using it — a config shape mismatch between a `configs/*.yaml` edit and its corresponding pydantic schema in `fnorollout/schemas/` will raise a `pydantic.ValidationError` at the very start of a run rather than failing later inside the training loop.

## Data Generation (GeophysicalFlows.jl)

Trajectory datasets are generated separately via Julia, not part of the `uv` environment:

```bash
cd fnorollout/julia/datagen
julia --project=. scripts/ns2d_data.jl   # intended to write a trajectory dataset under data/
```

`ns2d_data.jl` is a work in progress — check it before assuming it produces a NetCDF trajectory file yet.

Move/symlink the resulting NetCDF file into `data/processed/` and point a `configs/data_sources/*.yaml` entry's `path` at it (matching its NetCDF variable name(s) in that source's `channels` list). Julia isn't on `PATH` in non-interactive shells by default — use `~/.juliaup/bin/julia` if `julia` isn't found.
