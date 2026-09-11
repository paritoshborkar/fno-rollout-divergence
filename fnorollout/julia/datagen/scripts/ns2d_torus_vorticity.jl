using CUDA
using FFTW, Printf, Random
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

# Runs one simulation from a fully-resolved config and writes the trajectory to
# output_path. Defined as a function (rather than top-level script code) so a sweep
# driver can `include` this file and call it repeatedly in a single Julia process,
# paying Julia/CUDA's JIT compilation cost once instead of once per sample.
function run_simulation(config::Dict, output_path::String)
    seed = Int(config["seed"])

    grid_resolution = Int(config["numerics"]["n"])            # 2D resolution = n^2
    domain_size = Float64(config["numerics"]["domain_size"])  # unit torus
    nu = Float64(config["numerics"]["nu"])                    # viscosity
    nu_order = Int(config["numerics"]["nu_order"])            # hyperviscosity order (nν)
    dt = Float64(config["numerics"]["dt"])                    # timestep
    stepper = String(config["numerics"]["stepper"])           # timestepper

    forcing_rate = Float64(config["forcing"]["rate"])              # ϵ: energy input rate by the forcing
    forcing_wavenumber = Float64(config["forcing"]["wavenumber"])  # forcing wavenumber (k in sin/cos(2π*k*(x+y)))

    grf_τ = Float64(config["initial_condition"]["grf_tau"])    # initial-condition Gaussian Random Field length scale
    grf_α = Float64(config["initial_condition"]["grf_alpha"])  # initial-condition Gaussian Random Field spectral slope

    n_snapshots = Int(config["output"]["n_snapshots"])         # number of vorticity snapshots to save
    save_interval = Float64(config["output"]["save_interval"]) # time units between saved snapshots

    # Gaussian random field: covariance ∝ (−Δ + τ²I)^(−α), on the unit torus
    function sample_gaussian_random_field(resolution; τ=grf_τ, α=grf_α, rng=Random.default_rng())
        k = fftfreq(resolution, resolution)                      # integer modes 0,1,…,N/2-1,-N/2,…,-1
        kx = reshape(k, resolution, 1)
        ky = reshape(k, 1, resolution)
        k2 = @. (2π * kx)^2 + (2π * ky)^2 # eigenvalues of −Δ

        σ = τ^(α - 1) # = 7^(3/2) for α=2.5, dim=2
        sqrt_eig = @. resolution^2 * sqrt(2.0) * σ * (k2 + τ^2)^(-α / 2)
        sqrt_eig[1, 1] = 0.0 # zero-mean field

        ŵ = sqrt_eig .* randn(rng, ComplexF64, resolution, resolution)
        return real(ifft(ŵ))
    end

    device = CUDA.functional() ? GPU() : CPU()
    println("Using $device to generate vorticity on the grid")

    # Set seed for reproducibility
    if device == CPU()
        Random.seed!(seed)
    else
        CUDA.seed!(seed)
    end

    grid = TwoDGrid(device; nx=grid_resolution, Lx=domain_size)
    x, y = gridpoints(grid)

    # Forcing function perpendicular to plane
    fh = rfft(@. forcing_rate * (sin(2π * forcing_wavenumber * (x + y)) + cos(2π * forcing_wavenumber * (x + y))))
    calcF!(Fh, sol, t, clock, vars, params, grid) = (@. Fh = fh; nothing)

    prob = TwoDNavierStokes.Problem(device; nx=grid_resolution, Lx=domain_size, ν=nu, nν=nu_order,
        dt=dt, stepper=stepper,
        calcF=calcF!, stochastic=false)

    # Initial condition: sample the GRF and set vorticity directly
    w_θ = sample_gaussian_random_field(grid_resolution)
    TwoDNavierStokes.set_ζ!(prob, w_θ)

    # Step, recording every `save_interval` time units, for `n_snapshots` snapshots
    T = n_snapshots * save_interval
    total_steps = round(Int, T / dt)
    save_every = round(Int, save_interval / dt)
    frames = Array{Float64}[]
    startwalltime = time()
    for i in 1:total_steps
        stepforward!(prob)
        if i % save_every == 0
            TwoDNavierStokes.updatevars!(prob)
            push!(frames, Array(prob.vars.ζ))

            log = @sprintf("step: %d/%d, t: %.3f, walltime: %.2f min", i, total_steps, i * dt, (time() - startwalltime) / 60)
            println(log)
            flush(stdout)
        end
    end

    ζ = cat(frames...; dims=3) # (nx, ny, n_frames)
    ζ = permutedims(ζ, (3, 1, 2)) # → (time, x, y), matching the xarray convention used elsewhere

    mkpath(dirname(output_path))
    times = collect(1:size(ζ, 1)) .* (save_every * dt)

    NCDataset(output_path, "c") do ds
        defDim(ds, "time", size(ζ, 1))
        defDim(ds, "x", grid_resolution)
        defDim(ds, "y", grid_resolution)
        defVar(ds, "time", times, ("time",))
        v = defVar(ds, "zeta", Float64, ("time", "x", "y"))
        v[:, :, :] = ζ
    end

    return output_path
end

function main()
    config_path = joinpath(@__DIR__, "..", "configs", "ns2d_torus_vorticity.toml")
    overrides = String[]
    for arg in ARGS
        if occursin('=', arg)
            push!(overrides, arg)
        else
            config_path = arg
        end
    end

    config = TOML.parsefile(config_path)
    println("Loaded simulation config from $config_path")
    for override in overrides
        apply_override!(config, override)
        println("Applied override: $override")
    end

    output_path = joinpath(@__DIR__, "..", "data", String(config["output"]["nc_filename"]))
    run_simulation(config, output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
