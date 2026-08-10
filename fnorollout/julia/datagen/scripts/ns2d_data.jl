using CUDA
using FFTW, Random
using GeophysicalFlows
using GeophysicalFlows: TwoDNavierStokes
using NCDatasets

# --- Simulation parameters ---
const GRID_RESOLUTION = 256
const DOMAIN_SIZE = 1.0 # unit torus
const ν = 1e-3 # viscosity
const NU_ORDER = 1 # hyperviscosity order (nν)
const DT = 1e-4 # timestep
const STEPPER = "ETDRK4" # Fourth order Runge Kutta solver

const FORCING_RATE = 0.1 # ϵ: energy input rate by the forcing
const FORCING_WAVENUMBER = 1 # forcing wavenumber (k in sin/cos(2π*k*(x+y)))

const GRF_τ = 7.0 # initial-condition Gaussian Random Field length scale
const GRF_α = 2.5 # initial-condition Gaussian Random Field spectral slope

const N_SNAPSHOTS = 50 # number of vorticity snapshots to save
const SAVE_INTERVAL = 1.0 # time units between saved snapshots

# Gaussian random field: covariance ∝ (−Δ + τ²I)^(−α), on the unit torus
function sample_gaussian_random_field(grid_resolution; τ=GRF_τ, α=GRF_α, rng=Random.default_rng())
    k = fftfreq(grid_resolution, grid_resolution)                      # integer modes 0,1,…,N/2-1,-N/2,…,-1
    kx = reshape(k, grid_resolution, 1)
    ky = reshape(k, 1, grid_resolution)
    k2 = @. (2π * kx)^2 + (2π * ky)^2 # eigenvalues of −Δ

    σ = τ^(α - 1) # = 7^(3/2) for α=2.5, dim=2
    sqrt_eig = @. grid_resolution^2 * sqrt(2.0) * σ * (k2 + τ^2)^(-α / 2)
    sqrt_eig[1, 1] = 0.0 # zero-mean field
    
    ŵ = sqrt_eig .* randn(rng, ComplexF64, grid_resolution, grid_resolution)
    return real(ifft(ŵ))
end


device = CUDA.functional() ? GPU() : CPU()
println("Using $device to generate vorticity on the grid")

grid = TwoDGrid(device; nx=GRID_RESOLUTION, Lx=DOMAIN_SIZE)
x, y = gridpoints(grid)

# Forcing function perpendicular to plane
fh = rfft(@. FORCING_RATE * (sin(2π * FORCING_WAVENUMBER * (x + y)) + cos(2π * FORCING_WAVENUMBER * (x + y))))
calcF!(Fh, sol, t, clock, vars, params, grid) = (@. Fh = fh; nothing)

prob = TwoDNavierStokes.Problem(device; nx=GRID_RESOLUTION, Lx=DOMAIN_SIZE, ν=ν, nν=NU_ORDER,
    dt=DT, stepper=STEPPER,
    calcF=calcF!, stochastic=false)

# Initial condition: sample the GRF and set vorticity directly
w_θ = sample_gaussian_random_field(GRID_RESOLUTION)
TwoDNavierStokes.set_ζ!(prob, w_θ)

# Step, recording every `SAVE_INTERVAL` time units, for `N_SNAPSHOTS` snapshots
T = N_SNAPSHOTS * SAVE_INTERVAL
save_every = round(Int, SAVE_INTERVAL / DT)
frames = Array{Float64}[]
for i in 1:round(Int, T / DT)
    stepforward!(prob)
    if i % save_every == 0
        TwoDNavierStokes.updatevars!(prob)
        push!(frames, Array(prob.vars.ζ))
    end
end


ζ = cat(frames...; dims=3) # (nx, ny, n_frames)
ζ = permutedims(ζ, (3, 1, 2)) # → (time, x, y), matching the xarray convention used elsewhere

output_path = joinpath(@__DIR__, "..", "data", "trajectory.nc")
mkpath(dirname(output_path))

times = collect(1:size(ζ, 1)) .* (save_every * DT)

NCDataset(output_path, "c") do ds
    defDim(ds, "time", size(ζ, 1))
    defDim(ds, "x", GRID_RESOLUTION)
    defDim(ds, "y", GRID_RESOLUTION)
    defVar(ds, "time", times, ("time",))
    v = defVar(ds, "zeta", Float64, ("time", "x", "y"))
    v[:, :, :] = ζ
end