#!/usr/bin/env julia
# profiling/bench_gradient.jl
#
# Micro-benchmark for the AD path: builds the same simulated workspace as
# profile_example.jl (NEID order 81, 50 obs, max_n_tel=max_n_star=2), then
# benchmarks ONE value_and_gradient! call and profiles 200 Adam iterations to
# check for Enzyme's runtime_generic_* fallback frames (see
# profiling/HANDOFF_PERF.md Step 1's pass/fail thresholds).
#
# Usage (from repo root):
#   OPENBLAS_NUM_THREADS=4 julia -t 4 profiling/bench_gradient.jl

using Pkg

const REPO = dirname(@__DIR__)
# Reuse the examples/ environment (see profile_example.jl for why a fresh
# from-scratch env is unsafe here: it can resolve incompatible transitive
# versions, e.g. of DataInterpolations). BenchmarkTools is added to
# examples/Project.toml as a dev/profiling-only dependency of that scratch
# environment, not of the package itself.
Pkg.activate(joinpath(REPO, "examples"))
Pkg.instantiate()

import StellarSpectraObservationFitting as SSOF
using JLD2, Statistics, KernelDensity, Distributions, StatsBase
using BenchmarkTools
using Profile  # stdlib; resolved via @stdlib regardless of Project.toml
Profile.init(n = 10^8, delay = 0.005)  # see profile_example.jl for why

include(joinpath(@__DIR__, "_setup.jl"))

println("Julia: $(VERSION), threads: $(Threads.nthreads())")
model, data_simulated, times = build_simulated_data()
model_new = build_initial_model(model, data_simulated, times)
mws = SSOF.ModelWorkspace(model_new, data_simulated)
aws = mws.total

println("typeof(mws): ", typeof(mws))
println("typeof(aws): ", typeof(aws))
flush(stdout)

println("─── value_and_gradient! (@btime) ─────────────────────────────")
b = @benchmark SSOF.value_and_gradient!($(aws.cache), $(aws.l), $(aws.θ))
show(stdout, MIME"text/plain"(), b)
println()
flush(stdout)

allocs = @allocated SSOF.value_and_gradient!(aws.cache, aws.l, aws.θ)
println("allocs/call (single @allocated sample): $allocs bytes")
flush(stdout)

println("─── 200 Adam iterations, profiled ────────────────────────────")
Profile.clear()
Profile.@profile for _ in 1:200
	SSOF.update!(aws)
end
Profile.print(; format=:flat, mincount=5)
