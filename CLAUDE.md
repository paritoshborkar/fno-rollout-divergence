# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`fno-rollout-divergence` is a research project studying rollout divergence in Fourier Neural Operators (FNOs). It uses PyTorch + `neuraloperator` for model implementation, Hydra + OmegaConf for experiment configuration, and Weights & Biases for experiment tracking.

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
| `neuraloperator` | FNO model implementations |
| `hydra-core` + `omegaconf` | Hierarchical config management for experiments |
| `wandb` | Experiment tracking and logging |
| `matplotlib` | Plotting/visualization |
| `xarray` + `netcdf4` | Reading NetCDF trajectory data (e.g. Oceananigans output) |

## Repository Structure

```
configs/                        # Hydra config groups
  config.yaml                    #   root: defaults list, hydra.run.dir, wandb settings
  data/default.yaml
  model/fno2d.yaml
  training/default.yaml
  experiment/baseline.yaml       #   +experiment=<name> override bundles (use `# @package _global_`)

fno_rollout_divergence/          # Python package (flat layout, no src/)
  data/                           #   dataset loading + preprocessing
  models/                         #   FNO construction from `model` config group
  train.py                        #   @hydra.main entrypoint

data/
  raw/, processed/                # gitignored except .gitkeep; processed/trajectory.nc is the
                                   # default dataset path referenced by configs/data/default.yaml

julia/oceananigans/               # separate Julia toolchain (juliaup) for generating trajectory
                                   # data with Oceananigans.jl; has its own Project.toml/Manifest.toml,
                                   # independent of the uv-managed Python environment
```

`fno_rollout_divergence/` is currently a bare skeleton (stub modules, `train.py`'s `main()` is a no-op) — check the actual file contents before assuming any training/data logic is implemented.

## Configuration

Hydra is the config system, configs live under `configs/` as shown above. Run scripts typically accept `+experiment=<name>` or `hydra.run.dir=...` overrides on the CLI, e.g.:

```bash
uv run python -m fno_rollout_divergence.train +experiment=baseline
uv run python -m fno_rollout_divergence.train wandb.mode=offline training.epochs=5
```

Hydra's `hydra.run.dir` (`outputs/<date>/<time>/`, gitignored) is the per-run working directory (`hydra.job.chdir: true`) — logs and training artifacts for a run should be written there rather than to a separate top-level directory. OmegaConf structured configs are used for type-safe config validation.

## Data Generation (Oceananigans.jl)

Trajectory datasets are generated separately via Julia, not part of the `uv` environment:

```bash
cd julia/oceananigans
julia --project=. scripts/generate_dataset.jl   # writes data/trajectory.nc there
```

Move/symlink the resulting NetCDF file into `data/processed/` to match `configs/data/default.yaml`'s `path`. Julia isn't on `PATH` in non-interactive shells by default — use `~/.juliaup/bin/julia` if `julia` isn't found.
