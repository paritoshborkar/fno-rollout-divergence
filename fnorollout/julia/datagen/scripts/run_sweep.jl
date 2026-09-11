# Runs every sample listed in a plan file (written by
# fnorollout/scripts/generate_ns2d_data.py) in a single Julia process, so package
# loading and CUDA/JIT kernel compilation happen once for the whole sweep instead of
# once per sample.
#
# Usage:
#   julia --project=. scripts/run_sweep.jl path/to/plan.toml

using Printf
using TOML

function format_elapsed(seconds::Real)
    seconds < 60 && return @sprintf("%.1fs", seconds)
    minutes, seconds = divrem(seconds, 60)
    return @sprintf("%dm %.1fs", minutes, seconds)
end

if length(ARGS) < 1
    error("usage: julia run_sweep.jl <plan.toml>")
end

sweep_plan = TOML.parsefile(ARGS[1])

# Defines run_simulation, apply_override!, etc. Guarded by `if abspath(PROGRAM_FILE) ==
# @__FILE__`, so including it here only defines those functions without also running
# its own standalone main().
include(joinpath(@__DIR__, sweep_plan["script"]))

base_config = TOML.parsefile(sweep_plan["config"])
samples = sweep_plan["samples"]
total = length(samples)

run_start = time()
for (i, sample) in enumerate(samples)
    config = deepcopy(base_config)
    for (key, value) in sample["overrides"]
        apply_override!(config, "$key=$value")
    end

    output_path = sample["output_path"]
    println("[$i/$total] overrides=$(sample["overrides"]) -> $output_path")

    sample_start = time()
    run_simulation(config, output_path)
    println("  sample done in $(format_elapsed(time() - sample_start))")
    flush(stdout)
end

println("Completed $total samples in $(format_elapsed(time() - run_start)) total")
