# Create class that provides statistics about vorticity trajectory (simulated or predicted)


import torch


class TrajectoryStatistics:
    def __init__(self, trajectory: torch.Tensor, dx: float = 1.0, dy: float = 1.0):
        self.trajectory = trajectory
        self.dx = dx
        self.dy = dy


class Vorticity2DTrajectoryStatistics(TrajectoryStatistics):
    def __init__(self, trajectory: torch.Tensor, dx: float = 1.0, dy: float = 1.0):
        """
        trajectory: torch Tensor tracking vorticity over a 2D grid. Shape: T x X x Y
        dx: grid spacing in the x direction
        dy: grid spacing in the y direction

        Spectral quantities (psd, velocity, kinetic energy) use angular wavenumbers
        based on dx/dy (both default to 1.0, i.e. unit grid spacing, which caps the
        Nyquist wavenumber at +-pi regardless of resolution). Passing a non-default
        dx/dy rescales the wavenumber (and therefore velocity/energy) axis
        accordingly.
        """
        super().__init__(trajectory=trajectory, dx=dx, dy=dy)
        self.kx, self.ky = self._wavenumbers()

    def mean_per_snapshot(self) -> torch.Tensor:
        """
        Spatial mean of vorticity at each timestep. Shape: (T,)
        """
        return self.trajectory.mean(dim=(-2, -1))

    def variance_per_snapshot(self) -> torch.Tensor:
        """
        Spatial variance of vorticity at each timestep. Shape: (T,)
        """
        return self.trajectory.var(dim=(-2, -1))

    def _wavenumbers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        kx, ky: 1D angular wavenumber axes, based on dx/dy
        """
        nx, ny = self.trajectory.shape[-2], self.trajectory.shape[-1]
        device = self.trajectory.device
        kx = 2 * torch.pi * torch.fft.fftfreq(n=nx, d=self.dx, device=device)
        ky = 2 * torch.pi * torch.fft.fftfreq(n=ny, d=self.dy, device=device)

        return kx, ky

    def psd_per_snapshot(
        self, n_bins: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the isotropic Power Spectral Density of vorticity per snapshot:
        |zeta_hat(k)|^2 radially binned over wavenumber magnitude |k|.

        Returns (k_bins, psd) where k_bins has shape (n_bins,) and psd has shape
        (T, n_bins).
        """
        zeta_hat = torch.fft.fft2(self.trajectory, dim=(-2, -1))
        power = zeta_hat.abs() ** 2

        Kx, Ky = torch.meshgrid(self.kx, self.ky, indexing="ij")
        k = torch.sqrt(Kx**2 + Ky**2)

        nx, ny = self.trajectory.shape[-2], self.trajectory.shape[-1]
        n_bins = n_bins or max(nx, ny) // 2
        bin_edges = torch.linspace(0, k.max(), n_bins + 1, device=k.device)
        bin_indices = torch.bucketize(k.flatten(), bin_edges[1:-1])

        power_flat = power.reshape(*power.shape[:-2], -1)
        psd = torch.zeros(*power.shape[:-2], n_bins, device=power.device)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.any():
                psd[..., b] = power_flat[..., mask].sum(dim=-1)

        k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return k_centers, psd

    def enstrophy_per_snapshot(self) -> torch.Tensor:
        """
        Enstrophy density (0.5 * mean(zeta^2)) at each timestep. Shape: (T,)
        """
        return 0.5 * (self.trajectory**2).mean(dim=(-2, -1))

    def total_enstrophy(self) -> torch.Tensor:
        """
        Time averaged enstrophy of the system
        """
        return self.enstrophy_per_snapshot().mean()

    def _psi_hat_per_snapshot(self) -> torch.Tensor:
        """
        Streamfunction in Fourier space, solved spectrally from vorticity
        (Delta psi = -zeta -> psi_hat = zeta_hat / k^2, zero mode forced to 0)
        """
        Kx, Ky = torch.meshgrid(self.kx, self.ky, indexing="ij")
        k2 = Kx**2 + Ky**2
        k2[0, 0] = 1.0  # avoid divide-by-zero; zero mode is forced to 0 below anyway

        zeta_hat = torch.fft.fft2(self.trajectory, dim=(-2, -1))
        psi_hat = zeta_hat / k2
        psi_hat[..., 0, 0] = 0.0

        return psi_hat

    def _streamfunction_per_snapshot(self) -> torch.Tensor:
        """Real-space streamfunction derived from vorticity. Shape: T x X x Y."""
        return torch.fft.ifft2(self._psi_hat_per_snapshot(), dim=(-2, -1)).real

    def velocity_per_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Velocity field (u, v) derived from vorticity via the streamfunction
        (Delta psi = -zeta, u = d(psi)/dy, v = -d(psi)/dx), computed spectrally.
        Each of shape T x X x Y.
        """
        psi_hat = self._psi_hat_per_snapshot()

        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat

        u = torch.fft.ifft2(u_hat, dim=(-2, -1)).real
        v = torch.fft.ifft2(v_hat, dim=(-2, -1)).real

        return u, v

    def mean_velocity_per_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Spatial mean of each velocity component at each timestep. Each shape: (T,)"""
        u, v = self.velocity_per_snapshot()
        return u.mean(dim=(-2, -1)), v.mean(dim=(-2, -1))

    def mean_speed_per_snapshot(self) -> torch.Tensor:
        """Spatial mean of |velocity| at each timestep. Shape: (T,)"""
        u, v = self.velocity_per_snapshot()
        return torch.sqrt(u**2 + v**2).mean(dim=(-2, -1))

    def energy_density_per_snapshot(self) -> torch.Tensor:
        """
        Mean kinetic energy density (0.5 * mean(u^2 + v^2)) at each timestep. Shape: (T,)
        """
        u, v = self.velocity_per_snapshot()
        return 0.5 * (u**2 + v**2).mean(dim=(-2, -1))

    def relative_l2_per_snapshot(
        self, baseline_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Relative L2 error against a baseline trajectory at each timestep:
        ||self - baseline|| / ||baseline||, spatial norm. Shape: (T,)
        """
        diff = (self.trajectory - baseline_trajectory).flatten(-2)
        base = baseline_trajectory.flatten(-2)

        return torch.linalg.norm(diff, dim=-1) / torch.linalg.norm(base, dim=-1)  # Default L2 norm

    def relative_l2(self, baseline_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Calculte L2 loss relative to a simulated baseline trajectory
        """
        return self.relative_l2_per_snapshot(baseline_trajectory).mean()

    def energy_drift_per_snapshot(
        self, baseline_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Difference in kinetic energy density between this trajectory and a baseline,
        at each timestep: KE_self(t) - KE_baseline(t). Shape: (T,)
        """
        baseline_stats = Vorticity2DTrajectoryStatistics(
            baseline_trajectory, dx=self.dx, dy=self.dy
        )
        return (
            self.energy_density_per_snapshot()
            - baseline_stats.energy_density_per_snapshot()
        )

    def energy_drift(self, baseline_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Root-mean-square kinetic energy drift over the trajectory relative to a baseline.
        """
        drift = self.energy_drift_per_snapshot(baseline_trajectory)
        return torch.sqrt((drift**2).mean())
