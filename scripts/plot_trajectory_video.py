"""
Render a NetCDF trajectory dataset as a video, one frame per time snapshot.
Outputs mp4 by default; falls back to gif if ffmpeg isn't available.

Usage:
    uv run python scripts/plot_trajectory_video.py data/raw/torus2d_trajectory.nc
    uv run python scripts/plot_trajectory_video.py data/raw/torus2d_trajectory.nc --out traj.mp4 --fps 15
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from torch import fft


def _select_writer(out_path: Path, fps: int) -> tuple[object, Path]:
    """
    Picks FFMpegWriter (mp4) by default, falling back to PillowWriter (gif) if ffmpeg
    isn't available. Returns the writer and out_path with its
    extension corrected to match (mp4/gif), overriding whatever was passed in.
    """
    if FFMpegWriter.isAvailable():
        return FFMpegWriter(fps=fps), out_path.with_suffix(".mp4")

    print("ffmpeg not found, falling back to gif output")
    return PillowWriter(fps=fps), out_path.with_suffix(".gif")


def _resolve_time_dim(field: xr.DataArray) -> str:
    """
    Datasets in this repo name the time dimension either "time" or "t".
    """
    for candidate in ("time", "t"):
        if candidate in field.dims:
            return candidate
    raise ValueError(f"no 'time' or 't' dimension found in dims {field.dims}")


def _spectrum_magnitude(snapshot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the (fftshifted) 2D magnitude spectrum of a single snapshot, along with
    its corresponding (fftshifted) wavenumber axes. Assumes unit grid spacing (dx=1,
    no domain size supplied here), so kx/ky are angular wavenumbers capped at the
    fixed Nyquist limit +-pi regardless of resolution.
    """
    nx, ny = snapshot.shape
    kx = fft.fftshift(fft.fftfreq(nx)) * 2 * torch.pi
    ky = fft.fftshift(fft.fftfreq(ny)) * 2 * torch.pi

    wave_fft = fft.fft2(snapshot, norm="forward")
    wave_fft = fft.fftshift(wave_fft)
    magnitude_spectrum = torch.abs(wave_fft)

    return kx, ky, magnitude_spectrum


def _radial_psd(
    kx: torch.Tensor, ky: torch.Tensor, magnitude_spectrum: torch.Tensor, n_bins: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Radially bins power |FFT|^2 over wavenumber magnitude |k| = sqrt(kx^2 + ky^2),
    giving an isotropic 1D power spectral density from a 2D magnitude spectrum (as
    returned by _magnitude_spectrum).

    Returns (k_bins, psd), each of shape (n_bins,).
    """
    k = torch.sqrt(kx.reshape(-1, 1) ** 2 + ky.reshape(1, -1) ** 2)
    power = magnitude_spectrum**2

    n_bins = n_bins or max(magnitude_spectrum.shape) // 2
    bin_edges = torch.linspace(0, k.max(), n_bins + 1)
    bin_indices = torch.bucketize(k.flatten(), bin_edges[1:-1])

    power_flat = power.flatten()
    psd = torch.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            psd[b] = power_flat[mask].sum()

    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return k_centers, psd


def render_trajectory_video(
    path: Path,
    channel: str,
    out_path: Path,
    fps: int = 10,
    cmap: str = "RdBu",
) -> Path:
    """
    Plots a trajectory dataset of a single scalar value like streamfunction or vorticity values
    tracked on a 2D grid, alongside its 2D FFT magnitude spectrum and radially-binned
    power spectral density.

    Creates a gif of the resulting evolution.
    """
    dataset = xr.open_dataset(path)
    field = dataset[channel]  # dims: (y, x, time) or (t, y, x)
    time_dim = _resolve_time_dim(field)
    n_frames = field.sizes[time_dim]

    vmin, vmax = float(field.min()), float(field.max())

    # Precompute snapshots and their magnitude spectra once, so `update` just
    # indexes into them instead of re-slicing xarray / re-running FFTs per frame,
    # and so the spectrum subplot gets a single fixed color scale across the video.
    snapshots = [torch.from_numpy(field.isel({time_dim: i}).values) for i in range(n_frames)]
    kx, ky, _ = _spectrum_magnitude(snapshots[0])
    spectra = [_spectrum_magnitude(s)[2] for s in snapshots]
    spectrum_vmin = min(s.min().item() for s in spectra)
    spectrum_vmax = max(s.max().item() for s in spectra)

    # Radially-binned PSD per snapshot, reusing the magnitude spectra already computed
    # above (same wavenumber convention, no re-run of the FFT).
    psd_floor = 1e-20
    k_bins, _ = _radial_psd(kx, ky, spectra[0])
    psds = [_radial_psd(kx, ky, s)[1] + psd_floor for s in spectra]
    psd_ymin = min(p.min().item() for p in psds)
    psd_ymax = max(p.max().item() for p in psds)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    image = ax1.imshow(snapshots[0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    fig.colorbar(image, ax=ax1, label=channel)
    title = ax1.set_title(f"{channel} Trajectory")

    # kx is derived from the snapshot's row (dim0) size, ky from its column (dim1)
    # size, but imshow plots dim0 along the Y-axis and dim1 along the X-axis, so the
    # extent (and labels) need ky first, then kx, to match.
    spectrum_extent = [ky.min().item(), ky.max().item(), kx.min().item(), kx.max().item()]
    spectrum_image = ax2.imshow(
        spectra[0].numpy(),
        cmap="viridis",
        vmin=spectrum_vmin,
        vmax=spectrum_vmax,
        origin="lower",
        extent=spectrum_extent,
    )
    fig.colorbar(spectrum_image, ax=ax2, label="|FFT|")
    ax2.set_title("Magnitude spectrum")
    ax2.set_xlabel("ky")
    ax2.set_ylabel("kx")

    (psd_line,) = ax3.plot(k_bins.numpy(), psds[0].numpy())
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(k_bins.min().item(), k_bins.max().item())
    ax3.set_ylim(psd_ymin, psd_ymax)
    ax3.set_title("Power spectral density")
    ax3.set_xlabel("|k|")
    ax3.set_ylabel("PSD")
    fig.tight_layout()

    def update(frame_index: int):
        image.set_data(snapshots[frame_index].numpy())
        title.set_text(f"{channel} at {time_dim}={float(field[time_dim][frame_index]):.2f}")
        spectrum_image.set_data(spectra[frame_index].numpy())
        psd_line.set_ydata(psds[frame_index].numpy())
        return image, title, spectrum_image, psd_line

    anim = FuncAnimation(fig, update, frames=n_frames)

    writer, out_path = _select_writer(out_path, fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to a trajectory NetCDF file")
    parser.add_argument("--channel", default="zeta", help="Variable name to render")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output video path (.mp4, or .gif if ffmpeg is unavailable)"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    args = parser.parse_args()

    out_path = args.out or args.path.with_name(f"{args.channel}.mp4")
    out_path = render_trajectory_video(args.path, args.channel, out_path, fps=args.fps)
    print(f"saved video to {out_path}")


if __name__ == "__main__":
    main()
