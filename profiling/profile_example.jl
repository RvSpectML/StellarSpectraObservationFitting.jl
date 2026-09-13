#!/usr/bin/env julia
# profiling/profile_example.jl
#
# Reproducible driver for the profiling work described in profiling/PROFILING.md
# and profiling/HANDOFF_PERF.md. Runs the same simulated NEID order-81 pipeline
# as examples/example.jl (50 observations, max_n_tel=max_n_star=2) and profiles
# each stage (calculate_initial_model, fit_regularization!, improve_model!)
# separately with Profile.@profile(mincount=5) flat+tree dumps.
#
# Usage (from repo root):
#   OPENBLAS_NUM_THREADS=4 julia -t 4 profiling/profile_example.jl [output_dir]
#
# Writes to output_dir/: timings.txt, profile_flat_<stage>.txt,
# profile_tree_<stage>.txt, and results.jld2 (rvs/rvs_σ/stage wall times, for
# round-off comparison of Step 1's before/after runs). Redirect stdout to
# capture run.log, e.g. `... > output_dir/run.log 2>&1`.

using Pkg, Dates

const REPO = dirname(@__DIR__)
outdir = length(ARGS) >= 1 ? ARGS[1] :
	joinpath(REPO, "profiling", "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(outdir)

# Reuse the examples/ environment (same one examples/example.jl activates) rather
# than resolving a fresh env from scratch: a from-scratch resolve is not pinned
# to the same DataInterpolations/TemporalGPs/StaticArrays versions as the
# Manifest here and can silently pick incompatible ones (e.g. DataInterpolations'
# `extrapolate=true` kwarg is version-specific).
Pkg.activate(joinpath(REPO, "examples"))
Pkg.instantiate()

import StellarSpectraObservationFitting as SSOF
using JLD2, Statistics, KernelDensity, Distributions, StatsBase
using Profile  # stdlib; resolved via @stdlib regardless of Project.toml

# Default buffer (1e6 samples) overflows silently on stages this long/deep
# (calculate_initial_model, fit_regularization each ran ~500-700s and overflowed
# it in practice) -- Profile.clear() does not resize it, so this must run once
# up front. Coarsen delay to 5ms so 10-15 min stages still fit comfortably.
Profile.init(n = 10^8, delay = 0.005)

include(joinpath(@__DIR__, "_setup.jl"))

timings_file = joinpath(outdir, "timings.txt")
open(timings_file, "w") do io
	println(io, "stage,wall_seconds")
end
function record_timing(name, twall)
	open(timings_file, "a") do io
		println(io, "$name,$twall")
	end
end

function profile_stage(f, name)
	flush(stdout)
	Profile.clear()
	result = nothing
	twall = @elapsed (result = Profile.@profile f())
	open(joinpath(outdir, "profile_flat_$name.txt"), "w") do io
		Profile.print(io; format=:flat, mincount=5)
	end
	open(joinpath(outdir, "profile_tree_$name.txt"), "w") do io
		Profile.print(io; format=:tree, mincount=5)
	end
	println("[$name] wall time: $twall s")
	flush(stdout)
	record_timing(name, twall)
	return result, twall
end

println("Julia: $(VERSION), threads: $(Threads.nthreads()), output: $outdir")
flush(stdout)

t_setup = @elapsed ((model, data_simulated, times) = build_simulated_data())
println("[setup_simulate_data] wall time: $t_setup s")
flush(stdout)
record_timing("setup_simulate_data", t_setup)

model_new, twall_cim = profile_stage(() -> build_initial_model(model, data_simulated, times), "calculate_initial_model")

t_mws = @elapsed (mws = SSOF.ModelWorkspace(model_new, data_simulated))
println("[ModelWorkspace_construction] wall time: $t_mws s")
flush(stdout)
record_timing("ModelWorkspace_construction", t_mws)

_, twall_reg = profile_stage(() -> SSOF.fit_regularization!(mws), "fit_regularization")

_, twall_imp = profile_stage(() -> SSOF.improve_model!(mws; iter=500, verbose=true, careful_first_step=true, speed_up=false), "improve_model")

rvs = rvs_σ = tel_s_σ = star_s_σ = nothing
try
	rvs, rvs_σ, tel_s_σ, star_s_σ = SSOF.estimate_σ_curvature(mws)
catch e
	println("estimate_σ_curvature failed (tracked open item, see profiling/HANDOFF_PERF.md): ", sprint(showerror, e))
	flush(stdout)
end

@save joinpath(outdir, "results.jld2") rvs rvs_σ tel_s_σ star_s_σ twall_cim twall_reg twall_imp

println("Done. Output in $outdir")
flush(stdout)
