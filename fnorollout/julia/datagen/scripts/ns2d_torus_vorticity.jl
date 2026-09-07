using CUDA
using FFTW, Random
using GeophysicalFlows
using GeophysicalFlows: TwoDNavierStokes
using NCDatasets
using TOML

# --- Simulation Parameters ---
# Any ARGS entry containing '=' is a dotted-key override (e.g. `numerics.dt=1e-5`),
# applied on top of the config file. A lone entry without '=' overrides the config path.
function parse_override_value(raw::AbstractString)
    lowered = lowercase(raw)
    lowered == "true" && return true
    lowered == "false" && return false
    int_val = tryparse(Int, raw)
    int_val !== nothing && return int_val
    float_val = tryparse(Float64, raw)
    float_val !== nothing && return float_val
    return raw
end

function apply_override!(config::Dict, override::AbstractString)
    key, raw_value = split(override, '='; limit=2)
    path = split(key, '.')
    node = config
    for k in path[1:end-1]
        node = get!(node, k, Dict{String,Any}())
    end
    node[path[end]] = parse_override_value(raw_value)
    return config
end

config_path = joinpath(@__DIR__, "..", "configs", "ns2d_torus_vorticity.toml")
overrides = String[]
for arg in ARGS
    if occursin('=', arg)
        push!(overrides, arg)
    else
        global config_path = arg
    end
end

config = TOML.parsefile(config_path)
println("Loaded simulation config from $config_path")
for override in overrides
    apply_override!(config, override)
    println("Applied override: $override")
end

const SEED = Int(config["seed"])

const GRID_RESOLUTION = Int(config["numerics"]["n"])            # 2D resolution = n^2
const DOMAIN_SIZE = Float64(config["numerics"]["domain_size"])  # unit torus
const ν = Float64(config["numerics"]["nu"])                     # viscosity
const NU_ORDER = Int(config["numerics"]["nu_order"])            # hyperviscosity order (nν)
const DT = Float64(config["numerics"]["dt"])                    # timestep
const STEPPER = String(config["numerics"]["stepper"])           # timestepper

const FORCING_RATE = Float64(config["forcing"]["rate"])              # ϵ: energy input rate by the forcing
const FORCING_WAVENUMBER = Float64(config["forcing"]["wavenumber"])  # forcing wavenumber (k in sin/cos(2π*k*(x+y)))

const GRF_τ = Float64(config["initial_condition"]["grf_tau"])    # initial-condition Gaussian Random Field length scale
const GRF_α = Float64(config["initial_condition"]["grf_alpha"])  # initial-condition Gaussian Random Field spectral slope

const N_SNAPSHOTS = Int(config["output"]["n_snapshots"])         # number of vorticity snapshots to save
const SAVE_INTERVAL = Float64(config["output"]["save_interval"]) # time units between saved snapshots
const NC_FILENAME = String(config["output"]["nc_filename"])      # output NetCDF filename

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

# Set seed for reproducibility
if device==CPU()
    ;
    Random.seed!(SEED);
else
    ;
    CUDA.seed!(SEED);
end

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

output_path = joinpath(@__DIR__, "..", "data", NC_FILENAME)
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
