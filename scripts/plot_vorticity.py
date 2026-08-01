"""Plot a vorticity field at a given timestamp from a generated trajectory NetCDF.

Usage:
    uv run python scripts/plot_vorticity.py data/processed/trajectory.nc 20.0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a trajectory NetCDF file")
    parser.add_argument("time", type=float, help="Timestamp to plot (nearest match)")
    parser.add_argument("--out", type=Path, default=None, help="Output image path")
    args = parser.parse_args()

    ds = xr.open_dataset(args.path)
    field = ds["zeta"].sel(time=args.time, method="nearest")

    field.plot(cmap="RdBu_r")
    plt.title(f"vorticity at t={float(field.time):.2f}")

    out_path = args.out or args.path.with_name(f"vorticity_t{args.time:.2f}.png")
    plt.savefig(out_path)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
