"""Render a NetCDF trajectory dataset as a gif, one frame per time snapshot.

Usage:
    uv run python scripts/plot_trajectory_video.py data/processed/torus2d_trajectory.nc
    uv run python scripts/plot_trajectory_video.py data/processed/torus2d_trajectory.nc --out traj.gif --fps 15
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter


def render_trajectory_video(
    path: Path,
    channel: str,
    out_path: Path,
    fps: int = 10,
    cmap: str = "RdBu",
) -> Path:
    """
    Plots a trajectory dataset of a single scalar value like streamfunction or vorticity values
    tracked on a 2D grid.

    Creates a gif of the resulting evolution.
    """
    dataset = xr.open_dataset(path)
    field = dataset[channel]  # dims: (y, x, time)

    vmin, vmax = float(field.min()), float(field.max())

    fig, ax = plt.subplots()
    image = ax.imshow(
        field.isel(time=0).values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower"
    )
    fig.colorbar(image, ax=ax, label=channel)
    title = ax.set_title(f"{channel} Trajectory")

    def update(frame_index: int):
        image.set_data(field.isel(time=frame_index).values)
        title.set_text(f"{channel} at t={float(field.time[frame_index]):.2f}")
        return image, title

    anim = FuncAnimation(
        fig, update, frames=field.sizes["time"]
    )  # TODO: Remove NetCDF dimension magic strings

    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a trajectory NetCDF file")
    parser.add_argument("--channel", default="zeta", help="Variable name to render")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output video path (.gif)"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()

    out_path = args.out or args.path.with_name(f"{args.channel}.gif")
    render_trajectory_video(args.path, args.channel, out_path, fps=args.fps)
    print(f"saved video to {out_path}")


if __name__ == "__main__":
    main()
