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
  data/default.yaml               #   selects named sources from data_sources/ for train/test, train/val split,
                                   #   and a `preprocessing` block (fields to normalize, normalizer `_target_`)
  data_sources/                   #   reusable named data source defs (local file, package, url); referenced from
                                   #   configs/data/*.yaml via Hydra defaults + `@` package overrides, e.g.
                                   #   `- /data_sources@sources.<name>: <name>`
  model/fno2d.yaml
  training/default.yaml           #   optimizer, scheduler, dataloader batch sizes, checkpoints (path, save_every,
                                   #   keep_best) — training_loss/eval_losses live in the loss config group, not here
  loss/                           #   standalone h1.yaml/l2.yaml wired together by default.yaml via Hydra defaults +
                                   #   `@` package overrides (same pattern as data_sources): `/loss@training_loss.h1: h1`,
                                   #   `/loss@eval_losses.h1: h1`, `/loss@eval_losses.l2: l2`. train.py reads
                                   #   config.loss.training_loss / config.loss.eval_losses (not config.training.*).
                                   #   train_loop() picks training_loss via `next(iter(...))` — only the first entry
                                   #   under training_loss is ever used, so adding a second one silently has no effect.
  eval/default.yaml               #   long-horizon rollout eval settings — not yet wired into any script
  experiment/baseline.yaml        #   +experiment=<name> override bundles (use `# @package _global_`)

fnorollout/                      # Python package (flat layout, no src/), imported as `fnorollout.*`
  data/                           #   TrajectoryDataset (multi-channel NetCDF loader) + library code (not
                                   #   entrypoints) imported by fnorollout/scripts/train.py
    data_utils.py                   #   loads datasets from configured sources (dispatched by DataSourceType),
                                     #   builds train/val/test dataloaders
    preprocessing.py                #   build_data_processor: instantiates+fits a normalizer (per data.preprocessing's
                                     #   `_target_`, e.g. UnitGaussianNormalizer.from_dataset) into a DataProcessor for
                                     #   neuralop's Trainer; save_data_processor persists its stats alongside a checkpoint
  models/                         #   FNO construction from the `model` config group (build_fno2d)
  schemas/                        #   pydantic models validating the composed config (Config, DataConfig,
                                   #   PreprocessingConfig, DataSource)
  scripts/                        #   CLI entrypoints only (things you run, not things you import)
    train.py                       #   @hydra.main entrypoint: builds model/optimizer/scheduler/Trainer/
                                    #   data_processor, runs training via neuralop's Trainer, logs to wandb
    generate_ns2d_data.py           #   sweeps simulation params of any julia/datagen/scripts/*.jl script;
                                    #   see Data Generation below
  julia/datagen/                  #   Julia toolchain (juliaup) for generating trajectory data with
                                   #   GeophysicalFlows.jl; own Project.toml/Manifest.toml, independent of
                                   #   the uv-managed Python environment
    configs/*.toml                   #   per-script simulation parameters, parsed with Julia's stdlib TOML (not
                                      #   Hydra) — see Data Generation below for the CLI override syntax
    scripts/qg_beta_turbulence.jl     #   SingleLayerQG beta-plane turbulence, writes a NetCDF via
                                       #   configs/qg_beta_turbulence.toml's [output] filename
    scripts/ns2d_torus_vorticity.jl   #   TwoDNavierStokes on a torus, writes a NetCDF via
                                       #   configs/ns2d_torus_vorticity.toml's [output] filename
    scripts/run_sweep.jl              #   generic driver: given a plan.toml (written by generate_ns2d_data.py),
                                       #   `include`s the target script and calls its run_simulation() once per
                                       #   sample in a single Julia process — see Data Generation below

scripts/                         # Top-level, NOT part of the fnorollout package — don't confuse with fnorollout/scripts/
  plot_trajectory_video.py         #   renders a NetCDF trajectory as an animated gif/mp4, one frame per time
                                    #   snapshot; auto-detects whether the time dim/coord is named "time" or "t"
  plot_vorticity.py                #   quick NetCDF vorticity plotting utility (expects a "time" coord)
  setup_remote.sh                   #   bootstrap script for a fresh GPU instance

notebooks/                       # Exploratory Jupyter notebooks (prototyping only, not the source of truth)

data/
  raw/, processed/                # gitignored except .gitkeep. Generated datasets currently land in and are read
                                   # directly from data/raw/ (e.g. configs/data_sources/torus2d_example.yaml's `path`
                                   # points at data/raw/torus2d_trajectory.nc) — data/processed/ isn't used yet
```

Data source `type` handling in `fnorollout/data/data_utils.py` only implements `DataSourceType.LOCAL` — `URL` and `PACKAGE` sources (e.g. `configs/data_sources/neuralop_darcy.yaml`) raise `NotImplementedError`. Check `fnorollout/data/data_utils.py` before assuming a non-local source actually loads.

## Configuration

Hydra is the config system, configs live under `configs/` as shown above. Run scripts typically accept `+experiment=<name>` or `hydra.run.dir=...` overrides on the CLI, e.g.:

```bash
uv run python -m fnorollout.scripts.train +experiment=baseline
uv run python -m fnorollout.scripts.train wandb.mode=offline training.epochs=5
```

Hydra's `hydra.run.dir` (`outputs/<date>/<time>/`, gitignored) is the per-run working directory (`hydra.job.chdir: true`) — logs and training artifacts (e.g. `training.checkpoints.path`, currently `checkpoints`) should be written there rather than to a separate top-level directory.

`fnorollout/scripts/train.py`'s `main()` validates the fully composed config against `fnorollout.schemas.configs.Config` (a pydantic model) before using it — a config shape mismatch between a `configs/*.yaml` edit and its corresponding pydantic schema in `fnorollout/schemas/` will raise a `pydantic.ValidationError` at the very start of a run rather than failing later inside the training loop.

## Data Generation (GeophysicalFlows.jl)

Trajectory datasets are generated separately via Julia, not part of the `uv` environment. Both `scripts/qg_beta_turbulence.jl` (SingleLayerQG beta-plane turbulence) and `scripts/ns2d_torus_vorticity.jl` (TwoDNavierStokes on a torus) are implemented and produce NetCDF files.

Each script is structured as a `run_simulation(config, output_path)` function (the actual physics + NetCDF write) plus a `main()` (CLI arg/config-loading, calls `run_simulation` once) plus a trailing `if abspath(PROGRAM_FILE) == @__FILE__; main(); end` guard. This is what lets `scripts/run_sweep.jl` (below) `include()` a script and reuse `run_simulation()` directly without also triggering its standalone-CLI `main()`.

### Single-sample runs

```bash
cd fnorollout/julia/datagen
julia --project=. scripts/qg_beta_turbulence.jl
```

`qg_beta_turbulence.jl` reads simulation parameters from a TOML file (default `configs/qg_beta_turbulence.toml`; Julia's stdlib `TOML`, unrelated to Hydra) instead of hardcoding them. Any CLI arg containing `=` is a dotted-key override applied on top of that file (mirroring Hydra's `key=value` overrides); a lone arg without `=` overrides which config file is loaded:

```bash
julia --project=. scripts/qg_beta_turbulence.jl numerics.nsteps=500 physics.beta=5.0
julia --project=. scripts/qg_beta_turbulence.jl configs/other_run.toml numerics.stepper=RK4
```

`qg_beta_turbulence.jl` writes its NetCDF with dims `(x, y, t)` and a `t` coordinate (xarray reads this back as `(t, y, x)`) — this differs from `ns2d_torus_vorticity.jl`'s and the existing `torus2d_trajectory.nc`'s `(y, x, time)`/`time` convention, which `fnorollout/data/datasets.py`'s `TrajectoryDataset` assumes. Reconcile the dim order and coordinate name before wiring a `qg_beta_turbulence.jl` output into a `configs/data_sources/*.yaml` entry, or `TrajectoryDataset`'s blind `.permute(2, 1, 0)` will silently scramble the axes rather than error.

Move/symlink the resulting NetCDF file into `data/raw/` and point a `configs/data_sources/*.yaml` entry's `path` at it (matching its NetCDF variable name(s) in that source's `channels` list). Julia isn't on `PATH` in non-interactive shells by default — use `~/.juliaup/bin/julia` if `julia` isn't found.

### Sweeping many samples (`fnorollout/scripts/generate_ns2d_data.py`)

```bash
uv run python -m fnorollout.scripts.generate_ns2d_data \
    ns2d_torus_vorticity.jl ns2d_torus_vorticity.toml \
    --sweep initial_condition.grf_tau=3.0,5.0,7.0,10.0 \
    --total-samples 20 \
    --output-dir ns2d_sweep
```

Script-agnostic — works with either `.jl`/`.toml` pair. `--sweep VARIABLE=VALUE[,VALUE...]` is repeatable across variables; `--total-samples` splits evenly across sweeps, cycling through a sweep's listed values (with a perturbed seed) if its share exceeds the number of values given. Omit `--sweep` entirely to keep the base config unchanged and vary only the seed across `--total-samples` runs.

Rather than spawning one `julia` subprocess per sample — which used to pay Julia/CUDA's JIT compilation cost (often ~1 minute) on *every* sample — it builds the sample list, writes it as `sweep_plan.toml` inside the output directory, and invokes `scripts/run_sweep.jl` **once**. That script `include()`s the target `.jl` file (defining but not auto-running it, per the guard above) and calls `run_simulation()` in a loop, so package loading and CUDA kernel JIT compilation happen once for the whole sweep rather than once per sample. `sweep_plan.toml` is left behind afterward as a record of exactly which config overrides produced each output file.

Output lands at `data/raw/<output-dir>/<script_stem>/<sweep_variable_or_"seed">/<value>_seed<N>.nc` — namespaced by script name so two different scripts sharing an `--output-dir` never commingle output.

**GPU note**: at the grid resolutions these configs currently default to (128–256), GPU and CPU take roughly the same wall-clock time — per-CUDA-kernel-launch overhead dominates over actual FFT/RK4 compute at this scale (confirmed empirically on `qg_beta_turbulence.jl` at n=128), so a GPU run not beating CPU isn't a sign anything's broken. The crossover where GPU parallelism actually wins tends to be much larger grids (512+).
