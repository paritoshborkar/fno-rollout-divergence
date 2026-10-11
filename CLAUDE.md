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
| `boto3` + `python-dotenv` | Cloudflare R2 (S3-compatible) upload/download of `data/raw/` files (`scripts/upload_to_bucket.py` / `download_from_bucket.py`) |

## Repository Structure

```
configs/                        # Hydra config groups
  config.yaml                    #   root: defaults list (data, model, training, loss, eval), hydra.run.dir
                                   #   (${utcnow:...}, UTC - see Configuration below), wandb settings
  data/default.yaml               #   selects named sources from data_sources/ for train/(optional)test, train/val
                                   #   split, a `trajectory` block (rollout_steps, n_input_snapshots), and a
                                   #   `preprocessing` block (fields to normalize, normalizer `_target_`)
  data_sources/                   #   reusable named data source defs (local file *or directory*, package, url);
                                   #   referenced from configs/data/*.yaml via Hydra defaults + `@` package
                                   #   overrides, e.g. `- /data_sources@sources.<name>: <name>`. A directory `path`
                                   #   is scanned recursively for *.nc files; an optional `filter` field is a
                                   #   regex (re.search) applied to each matched filename to include/exclude
                                   #   samples, e.g. qg_beta_sweep.yaml's `filter: '^(?!15|1).*'` excludes the
                                   #   beta=15/beta=1 sweep values from a directory of beta-swept trajectories.
                                   #   A directory path is NOT subdirectory-aware beyond the regex - pointing it
                                   #   at a parent that contains both a training pool and a held-out test/
                                   #   subdirectory can silently leak test trajectories into training if the
                                   #   filter doesn't account for that.
  model/fno2d.yaml                #   in_channels is NOT automatically kept in sync with data.trajectory's
                                   #   n_input_snapshots (see fnorollout/data/datasets.py below) - must currently
                                   #   be set by hand to n_input_snapshots * len(channels)
  training/default.yaml           #   optimizer, scheduler, dataloader batch sizes (train/val/test), artifacts
                                   #   (path, save_every) — training_loss/eval_losses live in the loss config
                                   #   group, not here
  loss/                           #   standalone h1.yaml/l2.yaml wired together by default.yaml via Hydra defaults +
                                   #   `@` package overrides (same pattern as data_sources): `/loss@training_loss.h1: h1`,
                                   #   `/loss@eval_losses.h1: h1`, `/loss@eval_losses.l2: l2`. train_fno2d.py reads
                                   #   config.loss.training_loss / config.loss.eval_losses (not config.training.*).
                                   #   train_loop() picks training_loss via `next(iter(...))` — only the first entry
                                   #   under training_loss is ever used, so adding a second one silently has no effect.
  eval/default.yaml               #   long-horizon rollout eval settings — not yet wired into any script
  training_experiments/           #   +training_experiments=<name> override bundles for train_fno2d.py
                                   #   (`# @package _global_`). Convention: each file sets its own `id` (a float,
                                   #   e.g. 1.0, 2.1 for sub-variants), used via `${id}` interpolation in both
                                   #   hydra.run.dir (outputs/training_experiment_<id>) and wandb.name - keep these
                                   #   two in sync by hand per file, nothing enforces it. `default.yaml` here is a
                                   #   stale leftover from the old single `configs/experiment/baseline.yaml` (still
                                   #   references `+experiment=baseline` and a `training.lr` key that doesn't match
                                   #   training/default.yaml's actual shape) - not part of the current pattern.
  rollout_experiments/             #   +rollout_experiments=<name> override bundles for rollout_fno2d.py. Each file
                                   #   sets its own `id` (this rollout experiment's id) plus a mandatory
                                   #   `training_experiment_id: ???` that must match the training run's own `id`
                                   #   (e.g. `+rollout_experiments=experiment_1 training_experiment_id=1.0`) -
                                   #   drives both `artifacts.path` (outputs/training_experiment_<training_experiment_id>)
                                   #   and `output_path`/`hydra.run.dir` (outputs/experiments/<training_experiment_id>_<id>/).
  rollout2d_config.yaml           #   root config for rollout_fno2d.py (own hydra.run.dir default, `${utcnow:...}`,
                                   #   independent of configs/config.yaml); artifacts.path and output_path are `???`
                                   #   (mandatory) - must come from a +rollout_experiments=<name> override or CLI

fnorollout/                      # Python package (flat layout, no src/), imported as `fnorollout.*`
  utils.py                        #   set_seeds; register_utcnow_resolver() registers the `${utcnow:<strftime>}`
                                   #   OmegaConf resolver (UTC, not local time) used by hydra.run.dir and
                                   #   experiment output paths - called once at import time by both train_fno2d.py
                                   #   and rollout_fno2d.py, guarded by OmegaConf.has_resolver so importing both
                                   #   in one process doesn't raise on double-registration
  data/                           #   NeuralopsTrajectoryDataset (multi-channel, multi-file NetCDF loader) +
                                   #   library code (not entrypoints) imported by fnorollout/scripts/*.py
    data_utils.py                    #   load_dataset_from_path discovers files for a source (single file, or every
                                      #   *.nc recursively under a directory, optionally regex-`filter`ed - see
                                      #   data_sources/ above) and raises if that leaves zero files; dispatched by
                                      #   DataSourceType in load_dataset_from_source. create_neuralop_dataloaders
                                      #   builds train/val dataloaders: if `data.test_sources` is configured, train
                                      #   uses all of `data.sources` and val uses all of `data.test_sources`
                                      #   (`data_config.get("test_sources")`, not `in`, so an explicit `test_sources:
                                      #   null` falls back cleanly); otherwise each source in `data.sources` is
                                      #   independently train/val-split (split_dataset_indices) and concatenated.
                                      #   val_loaders is a dict keyed by grid resolution (shape[-1] of a sample's
                                      #   "x"), matching neuralop's Trainer `test_loaders` shape - sources sharing a
                                      #   resolution silently collide under the same key, and an empty val split
                                      #   (e.g. split.val=0) correctly yields `{}` rather than a 0-sample loader
                                      #   (which would crash Trainer.evaluate with a ZeroDivisionError).
    datasets.py                      #   NeuralopsTrajectoryDataset(file_paths, channel_names, rollout_steps,
                                      #   n_input_snapshots, channel_dim): loads each NetCDF file as one trajectory
                                      #   (T x C x Y x X). `"x"` is the window of n_input_snapshots consecutive
                                      #   snapshots ending at (and including) a sample's time_index, reshaped to
                                      #   (n_input_snapshots * C, Y, X) - i.e. prior snapshots are folded into the
                                      #   channel dimension, not a separate batch/time dim - so model.in_channels
                                      #   must be n_input_snapshots * len(channels) (see model/fno2d.yaml above).
                                      #   `"y"` is the single snapshot at time_index + rollout_steps.
    preprocessing.py                 #   build_data_processor: instantiates+fits a normalizer (per data.preprocessing's
                                      #   `_target_`, e.g. UnitGaussianNormalizer.from_dataset) into a DataProcessor for
                                      #   neuralop's Trainer; save_data_processor persists its stats alongside a checkpoint
  models/                         #   FNO construction from the `model` config group (build_fno2d)
  stats/                          #   trajectory_statistics.py: Vorticity2DTrajectoryStatistics - stats/metrics
                                   #   over a T x Y x X vorticity trajectory (mean/variance/PSD/enstrophy/velocity
                                   #   via spectral streamfunction solve/relative L2/energy drift/RMS/Wasserstein
                                   #   distance vs. a baseline trajectory) - used by fnorollout/scripts/compare_vorticities.py
  schemas/                        #   pydantic models validating the composed config: Config/DataConfig/
                                   #   PreprocessingConfig (configs/config.yaml, used by train_fno2d.py) and
                                   #   Rollout2DConfig (configs/rollout2d_config.yaml, used by rollout_fno2d.py),
                                   #   plus DataSource/LocalDataSource (configs/data_sources/*.yaml)
  scripts/                        #   CLI entrypoints only (things you run, not things you import)
    train_fno2d.py                  #   @hydra.main entrypoint (configs/config.yaml): builds model/optimizer/
                                     #   scheduler/Trainer/data_processor, runs training via neuralop's Trainer,
                                     #   saves pre/post-run artifacts (config.yaml, data_processor, model, optimizer,
                                     #   scheduler state dicts) to training.artifacts.path, logs to wandb
    rollout_fno2d.py                 #   @hydra.main entrypoint (configs/rollout2d_config.yaml): loads a previous
                                      #   training run's saved config + data_processor + model from artifacts.path,
                                      #   then reads n_input_snapshots back off that saved config (not a rollout-side
                                      #   knob - it's a property of how the model was trained). get_initial_snapshots
                                      #   slices the first n_input_snapshots raw frames (T x C x H x W); rollout()
                                      #   normalizes them once (in_normalizer), splits them into individual
                                      #   (1, C, H, W) snapshots, and - before the loop AND after every step -
                                      #   re-concatenates the *last* n_input_snapshots of them along the channel dim
                                      #   to form the next (1, n_input_snapshots*C, H, W) model input, sliding the
                                      #   window forward one snapshot at a time. The model's own output (back to
                                      #   out_normalizer only once, at the very end) is always a single C-channel
                                      #   snapshot, never pre-merged - only the window-builder does that. Saves the
                                      #   predicted trajectory as a NetCDF to output_path.
    compare_vorticities.py           #   loads a predicted + a real/simulated vorticity NetCDF trajectory, computes
                                      #   Vorticity2DTrajectoryStatistics for both, and plots/prints comparative
                                      #   metrics (enstrophy, energy density, energy drift, Wasserstein distance vs.
                                      #   RMS vorticity across early/mid/late rollout windows)
    generate_ns2d_data.py           #   sweeps simulation params of any julia/datagen/scripts/*.jl script;
                                    #   see Data Generation below
  julia/datagen/                  #   Julia toolchain (juliaup) for generating trajectory data with
                                   #   GeophysicalFlows.jl; own Project.toml/Manifest.toml, independent of
                                   #   the uv-managed Python environment
    configs/*.toml                   #   per-script simulation parameters, parsed with Julia's stdlib TOML (not
                                      #   Hydra) — see Data Generation below for the CLI override syntax
    scripts/qg_beta_forced.jl         #   SingleLayerQG beta-plane forced turbulence, writes a NetCDF via
                                       #   configs/qg_beta_forced.toml's [output] filename
    scripts/ns2d_torus_vorticity.jl   #   TwoDNavierStokes on a torus, writes a NetCDF via
                                       #   configs/ns2d_torus_vorticity.toml's [output] filename
    scripts/run_sweep.jl              #   generic driver: given a plan.toml (written by generate_ns2d_data.py),
                                       #   `include`s the target script and calls its run_simulation() once per
                                       #   sample in a single Julia process — see Data Generation below

scripts/                         # Top-level, NOT part of the fnorollout package — don't confuse with fnorollout/scripts/
  plot_trajectory_video.py         #   renders a NetCDF trajectory as an animated gif/mp4 (one frame per time
                                    #   snapshot) with FFT magnitude-spectrum and radially-binned PSD subplots
                                    #   alongside the field; auto-detects whether the time dim/coord is named
                                    #   "time" or "t"; prefers an mp4 writer, falls back to gif if unavailable
  plot_vorticity.py                #   quick NetCDF vorticity plotting utility (expects a "time" coord)
  upload_to_bucket.py               #   uploads a file or directory under data/raw/ to a Cloudflare R2 bucket
                                     #   (boto3 S3-compatible client); skips files that already exist remotely
  download_from_bucket.py           #   downloads a file or directory (by S3 key prefix) from the R2 bucket into
                                     #   data/raw/; skips files that already exist locally. Both scripts read R2
                                     #   credentials (account id, access key, secret, bucket name) from a
                                     #   `dev_no_docker.env` file via python-dotenv - gitignored (`*.env`), never
                                     #   commit real credentials
  setup_remote.sh                   #   bootstrap script for a fresh GPU instance; runs ON the remote instance
                                     #   itself (curl | bash), so it has no access to local secrets like
                                     #   dev_no_docker.env - that has to be scp'd over separately, from the local
                                     #   machine, as its own step

notebooks/                       # Exploratory Jupyter notebooks (prototyping only, not the source of truth)

data/
  raw/, processed/                # gitignored except .gitkeep. Generated datasets currently land in and are read
                                   # directly from data/raw/ (e.g. configs/data_sources/torus2d_example.yaml's `path`
                                   # points at data/raw/torus2d_trajectory.nc) — data/processed/ isn't used yet
```

Data source `type` handling in `fnorollout/data/data_utils.py` only implements `DataSourceType.LOCAL` — `URL` and `PACKAGE` sources (e.g. `configs/data_sources/neuralop_darcy.yaml`) raise `NotImplementedError`. Check `fnorollout/data/data_utils.py` before assuming a non-local source actually loads.

## Configuration

Hydra is the config system, configs live under `configs/` as shown above. `train_fno2d.py` typically accepts `+training_experiments=<name>` and `rollout_fno2d.py` accepts `+rollout_experiments=<name>` (or any `hydra.run.dir=...`/plain `key=value` override) on the CLI, e.g.:

```bash
uv run python -m fnorollout.scripts.train_fno2d
uv run python -m fnorollout.scripts.train_fno2d +training_experiments=experiment_1
uv run python -m fnorollout.scripts.train_fno2d wandb.mode=offline training.epochs=5

uv run python -m fnorollout.scripts.rollout_fno2d +rollout_experiments=experiment_1 training_experiment_id=1.0
```

Both entrypoints register a `${utcnow:<strftime pattern>}` OmegaConf resolver (`fnorollout.utils.register_utcnow_resolver`, UTC time) at import time, and `hydra.run.dir` is built from it (e.g. `configs/config.yaml`'s `outputs/${utcnow:%Y-%m-%d}/${utcnow:%H-%M-%S}`) rather than Hydra's own local-time `${now:...}`. With `hydra.job.chdir: true`, this resolved directory is the per-run working directory — logs and training artifacts (e.g. `training.artifacts.path`, currently `artifacts`) should be written there rather than to a separate top-level directory. A `+training_experiments=<name>`/`+rollout_experiments=<name>` override typically replaces this with a fixed, experiment-specific directory instead (e.g. `outputs/training_experiment_${id}`) — unlike the timestamped default, a fixed directory collides/overwrites across repeated runs unless the id itself is varied.

`fnorollout/scripts/train_fno2d.py`'s `main()` validates the fully composed config against `fnorollout.schemas.configs.Config` (a pydantic model) before using it — a config shape mismatch between a `configs/*.yaml` edit and its corresponding pydantic schema in `fnorollout/schemas/` will raise a `pydantic.ValidationError` at the very start of a run rather than failing later inside the training loop. `rollout_fno2d.py` validates against `fnorollout.schemas.rollout_config.Rollout2DConfig` the same way.

## Data Generation (GeophysicalFlows.jl)

Trajectory datasets are generated separately via Julia, not part of the `uv` environment. Both `scripts/qg_beta_forced.jl` (SingleLayerQG beta-plane forced turbulence) and `scripts/ns2d_torus_vorticity.jl` (TwoDNavierStokes on a torus) are implemented and produce NetCDF files.

Each script is structured as a `run_simulation(config, output_path)` function (the actual physics + NetCDF write) plus a `main()` (CLI arg/config-loading, calls `run_simulation` once) plus a trailing `if abspath(PROGRAM_FILE) == @__FILE__; main(); end` guard. This is what lets `scripts/run_sweep.jl` (below) `include()` a script and reuse `run_simulation()` directly without also triggering its standalone-CLI `main()`.

### Single-sample runs

```bash
cd fnorollout/julia/datagen
julia --project=. scripts/qg_beta_forced.jl
```

`qg_beta_forced.jl` reads simulation parameters from a TOML file (default `configs/qg_beta_forced.toml`; Julia's stdlib `TOML`, unrelated to Hydra) instead of hardcoding them. Any CLI arg containing `=` is a dotted-key override applied on top of that file (mirroring Hydra's `key=value` overrides); a lone arg without `=` overrides which config file is loaded:

```bash
julia --project=. scripts/qg_beta_forced.jl numerics.nsteps=500 physics.beta=5.0
julia --project=. scripts/qg_beta_forced.jl configs/other_run.toml numerics.stepper=RK4
```

Both scripts write their NetCDF with Julia dims `(x, y, t)` and a `t` coordinate — xarray/netCDF4 reads that reversed (Julia column-major vs. Python row-major) as `(t, y, x)`, i.e. already `(Time, Y, X)` with Y as rows and X as columns. `fnorollout/data/datasets.py`'s `NeuralopsTrajectoryDataset` relies on this and does no axis permutation — it just converts each file's `.values` straight to a tensor. Any new datagen script must follow the same `(x, y, t)` Julia dim order (verify with an asymmetric test field if unsure — square grids can't reveal an axis swap) or `NeuralopsTrajectoryDataset` will silently read it with axes scrambled rather than erroring.

Move/symlink the resulting NetCDF file into `data/raw/` and point a `configs/data_sources/*.yaml` entry's `path` at it (matching its NetCDF variable name(s) in that source's `channels` list) — or at the containing directory if you want every `*.nc` file under it loaded as a separate trajectory (optionally narrowed with `filter`, see Repository Structure above). Julia isn't on `PATH` in non-interactive shells by default — use `~/.juliaup/bin/julia` if `julia` isn't found.

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

Output lands at `data/raw/<script_stem>/<output-dir>/<sweep_variable_or_"seed">/<value>_seed<N>.nc` — namespaced by script name so two different scripts sharing an `--output-dir` never commingle output. A sweep variable's dots are replaced with underscores in the directory name (e.g. sweeping `physics.beta` lands under `.../physics_beta/<value>_seed<N>.nc`).

**GPU note**: at the grid resolutions these configs currently default to (128–256), GPU and CPU take roughly the same wall-clock time — per-CUDA-kernel-launch overhead dominates over actual FFT/RK4 compute at this scale (confirmed empirically on `qg_beta_forced.jl` at n=128), so a GPU run not beating CPU isn't a sign anything's broken. The crossover where GPU parallelism actually wins tends to be much larger grids (512+).
