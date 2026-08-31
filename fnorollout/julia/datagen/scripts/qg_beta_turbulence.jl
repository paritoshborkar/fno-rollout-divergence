# From GeophysicalFlows SingleLayer QG Beta-plane turbulence example: https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/literated/singlelayerqg_betaforced/

using GeophysicalFlows, CUDA, Random, Printf, JLD2, Statistics, NCDatasets, FFTW, TOML

using Statistics: mean
using LinearAlgebra: ldiv!

parsevalsum = FourierFlows.parsevalsum


device = CUDA.functional() ? GPU() : CPU()
println("Using $device to generate vorticity on the grid")

# --- Simulation Parameters ---
config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "qg_beta_turbulence.toml")
config = TOML.parsefile(config_path)
println("Loaded simulation config from $config_path")

const SEED = Int(config["seed"])

# Numerical and Timestepping parameters
const n = Int(config["numerics"]["n"])                        # 2D resolution = n^2
const stepper = String(config["numerics"]["stepper"])         # timestepper
const dt = Float64(config["numerics"]["dt"])                  # timestep
const nsteps = Int(config["numerics"]["nsteps"])               # total number of time-steps
const save_substeps = Int(config["numerics"]["save_substeps"]) # number of timesteps after which output is saved

# Physical parameters
L = Float64(config["physics"]["Lx_over_pi"]) * π  # domain size
β = Float64(config["physics"]["beta"])             # planetary Potential Vorticity gradient
μ = Float64(config["physics"]["mu"])               # bottom drag


# Forcing
forcing_wavenumber = Float64(config["forcing"]["wavenumber_coefficient"]) * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth = Float64(config["forcing"]["bandwidth_coefficient"]) * 2π/L    # the width of the forcing spectrum, `δ_f`
ε = Float64(config["forcing"]["epsilon"])                                          # energy input rate by the forcing

# Output paths
filename = "singlelayerqg_forcedbeta.jld2"
plotpath = "./plots_forcedbetaturbulence"
plotname = "snapshots"
filepath = joinpath(".", filename)

grid = TwoDGrid(device; nx=n, Lx=L)
K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
CUDA.@allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0       # normalize forcing to inject energy at rate ε 

# Set seed for reproducibility
if device==CPU()
    ;
    Random.seed!(SEED);
else
    ;
    CUDA.seed!(SEED);
end

# Calculates forcing at every timestep
function calcF!(Fh, sol, t, clock, vars, params, grid)
    randn!(Fh)
    @. Fh *= sqrt(forcing_spectrum) / sqrt(clock.dt)
    return nothing
end


# --- Problem Setup ---
prob = SingleLayerQG.Problem(device; nx=n, Lx=L, β, μ, dt, stepper,
    calcF=calcF!, stochastic=true)

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly


# --- Initial Condition: Fluid at rest ---
SingleLayerQG.set_q!(prob, device_array(device)(zeros(grid.nx, grid.ny)))


energy = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
enstrophy = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [energy, enstrophy] # A list of Diagnostics types passed to "stepforward!" will be updated every timestep.


if isfile(filepath); rm(filepath); end
if !isdir(plotpath); mkdir(plotpath); end


# --- Create solver output ---
get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution

function get_u(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  @. vars.qh = sol

  SingleLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  ldiv!(vars.u, grid.rfftplan, -im * grid.l .* vars.ψh)

  return Array(vars.u)
end

output = Output(prob, filepath, (:qh, get_sol), (:u, get_u))
saveproblem(output)
saveoutput(output)


# --- Timestepping the problem ---
startwalltime = time()

while clock.step < nsteps
  if clock.step % 50save_substeps == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    clock.step, clock.t, cfl, energy.data[energy.i], enstrophy.data[enstrophy.i], (time()-startwalltime)/60)

    println(log)
  end

  stepforward!(prob, diags, save_substeps)
  SingleLayerQG.updatevars!(prob)

  if clock.step % save_substeps == 0
    saveoutput(output)
  end
end

savediagnostic(energy, "energy", output.path)
savediagnostic(enstrophy, "enstrophy", output.path)



# --- Save to NetCDF5 file ---

infile  = filepath
outfilename = "singlelayerqg_forcedbeta.nc"

f = jldopen(infile, "r")
nx = f["grid/nx"];  ny = f["grid/ny"]
Lx = f["grid/Lx"];  Ly = f["grid/Ly"]
xg = collect(Float64, f["grid/x"])
yg = collect(Float64, f["grid/y"])

iters = sort(parse.(Int, keys(f["snapshots/t"])))   # keys arrive unsorted
t  = Float64[f["snapshots/t/$i"] for i in iters]
nt = length(iters)

ζ = Array{Float32}(undef, nx, ny, nt)
for (s, i) in enumerate(iters)
    qh = f["snapshots/qh/$i"]              # == ζh for barotropic
    ζ[:, :, s] .= Float32.(irfft(qh, nx))
end
close(f)

ds = NCDataset(outfilename, "c")

# Julia (column-major) dim order: x fastest. xarray/netCDF4 will see (t, y, x).
defDim(ds, "x", nx)
defDim(ds, "y", ny)
defDim(ds, "t", nt)          # use Inf here if you want an unlimited/appendable time axis

# coordinate variables sharing dim names → xarray auto-recognizes them as coords
cx = defVar(ds, "x", Float64, ("x",)); cx[:] = xg
cy = defVar(ds, "y", Float64, ("y",)); cy[:] = yg
ct = defVar(ds, "t", Float64, ("t",)); ct[:] = t
ct.attrib["long_name"] = "model time"

v = defVar(ds, "zeta", Float32, ("x", "y", "t");
           deflatelevel = 4,
           chunksizes   = (nx, ny, 1))     # one snapshot per chunk; given in Julia dim order
v[:, :, :] = ζ
v.attrib["long_name"] = "relative vorticity"

ds.attrib["beta"]    = β
ds.attrib["mu"]      = μ
ds.attrib["epsilon"] = ε
ds.attrib["k_f"]     = forcing_wavenumber
ds.attrib["dt"]      = dt
ds.attrib["Lx"] = Lx;  ds.attrib["Ly"] = Ly
ds.attrib["axis_order"] = "zeta read as (t, y, x) by xarray/netCDF4"
# also stamp git SHA + config hash + dataset hash here for the reproducibility triple

close(ds)

