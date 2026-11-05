"""
Plot rollout comparison metrics saved by fnorollout.rollout_utils.save_rollout_metrics
(a rollout_metrics.json file written alongside a predicted trajectory NetCDF).

Usage:
    uv run python -m fnorollout.scripts.plot_rollout_metrics --metrics-path outputs/.../rollout_metrics.json
    uv run python -m fnorollout.scripts.plot_rollout_metrics --metrics-path outputs/.../rollout_metrics.json --last-timestep-index 200
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> dict:
    with open(metrics_path) as f:
        return json.load(f)


def plot_metrics_over_time(metrics: dict, last_timestep_index: int | None = None) -> None:
    """
    last_timestep_index: if given, only snapshots up to and including this index
    (into metrics_over_time's "t"/per-timestep series) are plotted - everything
    after it is dropped before plotting.
    """
    end = None if last_timestep_index is None else last_timestep_index + 1
    t = metrics["metrics_over_time"]["t"][:end]
    divergence_step = metrics.get("divergence_step")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def _plot_comparative(ax, key: str, title: str):
        series = metrics["metrics_over_time"][key]
        ax.plot(t, series["predicted"][:end], color="b", label="predicted")
        ax.plot(t, series["real"][:end], color="r", label="real")
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.legend()

    _plot_comparative(axes[0], "enstrophy", "Enstrophy per snapshot")
    _plot_comparative(axes[1], "energy_density", "Energy density per snapshot")

    axes[2].plot(t, metrics["metrics_over_time"]["relative_l2"][:end], color="g")
    axes[2].set_title("Relative L2 (predicted vs real)")
    axes[2].set_xlabel("t")

    axes[3].plot(t, metrics["metrics_over_time"]["h1_loss"][:end], color="purple")
    axes[3].set_title("H1 loss (predicted vs real)")
    axes[3].set_xlabel("t")

    axes[4].plot(t, metrics["metrics_over_time"]["wasserstein_1d"][:end], color="orange")
    axes[4].set_title("Wasserstein-1 distance (predicted vs real, per snapshot)")
    axes[4].set_xlabel("t")

    if divergence_step is not None and divergence_step < len(t):
        for ax in axes[:5]:
            ax.axvline(
                t[divergence_step], color="black", linestyle="--", label="divergence"
            )

    axes[5].axis("off")
    summary_lines = [
        f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in metrics["summary"].items()
    ]
    if divergence_step is not None:
        summary_lines.insert(0, f"divergence_step: {divergence_step}")
    axes[5].text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        family="monospace",
        fontsize=8,
        transform=axes[5].transAxes,
    )

    fig.tight_layout()
    plt.show()


def main(cmd_args) -> None:
    metrics = load_metrics(Path(cmd_args.metrics_path))
    plot_metrics_over_time(metrics, last_timestep_index=cmd_args.last_timestep_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path", required=True, help="Path to a saved rollout_metrics.json"
    )
    parser.add_argument(
        "--last-timestep-index",
        type=int,
        default=None,
        help="If given, only plot snapshots up to and including this index",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
