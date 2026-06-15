#!/usr/bin/env julia
# benchmark_ad.jl
#
# Benchmarks the AD backend via the public SSOF API.
# Run on `master` (Nabla) and on `try_enzyme` (Enzyme) to compare.
#
# Usage (run from the repo root):
#   julia benchmark_ad.jl
#
# Results are written to benchmark_results_<branch>.txt.
# Structured as three focused benchmarks:
#   1. ModelWorkspace construction  -- one-time AD rule compilation cost
#   2. Adam update!                 -- steady-state per-step gradient cost (2 warmup + 10 timed)
#   3. finalize_scores!             -- score-only Optim gradient cost (1st and 2nd call)
#                                      wrapped in try/catch; skipped if Optim v1.11 bug fires

using Pkg

const REPO = dirname(abspath(@__FILE__))
bench_env = mktempdir(; cleanup=true)
Pkg.activate(bench_env)
Pkg.develop(PackageSpec(path=REPO); io=devnull)
Pkg.add(["JLD2", "Statistics", "Printf", "Dates"]; io=devnull)

import StellarSpectraObservationFitting as SSOF
using JLD2
using Statistics
using Printf
using Dates

branch = strip(readchomp(`git -C $REPO rev-parse --abbrev-ref HEAD`))
commit = strip(readchomp(`git -C $REPO rev-parse --short HEAD`))
outfile = joinpath(REPO, "benchmark_results_$(replace(branch, '/' => '-')).txt")

function tee(io, msg)
    print(msg)
    flush(stdout)
    print(io, msg)
    flush(io)
end

open(outfile, "w") do io
    header = """
================================================================
Branch : $branch ($commit)
Julia  : $(VERSION)
Date   : $(now())
================================================================
"""
    tee(io, header)

    data_dir = joinpath(REPO, "examples", "data")
    @load joinpath(data_dir, "results.jld2") model
    @load joinpath(data_dir, "data.jld2") data
    tee(io, "Loaded model ($(typeof(model))) and data ($(typeof(data)))\n\n")

    # ─── 1. ModelWorkspace construction (= AD rule compilation) ─────────────
    # One-time cost: ∇(l) JIT-warmup (Nabla) or prepare_gradient (Enzyme).
    # 2nd construction reuses the cached rule → reveals true caching benefit.
    tee(io, "─── 1. ModelWorkspace construction (AD compile) ─────────────────\n")
    GC.gc(); GC.enable(false)
    t1a = @elapsed mws = SSOF.ModelWorkspace(model, data)
    GC.enable(true); GC.gc(); GC.enable(false)
    t1b = @elapsed mws2 = SSOF.ModelWorkspace(model, data)
    GC.enable(true)
    tee(io, @sprintf("  1st construction : %7.2f s\n", t1a))
    tee(io, @sprintf("  2nd construction : %7.2f s\n", t1b))
    tee(io, "\n")

    # ─── 2. Adam update! — steady-state gradient evaluation ─────────────────
    # Accesses mws.total (AdamSubWorkspace) directly to bypass finalize_scores!.
    # Each update! = one value_and_gradient! + Adam step.
    # 2 warmup calls ensure the Julia method cache is hot before timing.
    tee(io, "─── 2. Adam update! — 10 samples after 2-step warmup ────────────\n")
    aws = mws.total
    for _ in 1:2; SSOF.update!(aws) end  # JIT warmup
    GC.gc()
    step_times = Vector{Float64}(undef, 10)
    for i in eachindex(step_times)
        GC.enable(false)
        step_times[i] = @elapsed SSOF.update!(aws)
        GC.enable(true)
    end
    tee(io, @sprintf("  median : %6.2f ms\n", 1e3 * median(step_times)))
    tee(io, @sprintf("  min    : %6.2f ms\n", 1e3 * minimum(step_times)))
    tee(io, @sprintf("  max    : %6.2f ms\n", 1e3 * maximum(step_times)))
    tee(io, @sprintf("  std    : %6.2f ms\n", 1e3 * std(step_times)))
    tee(io, "\n")

    # ─── 3. finalize_scores! — first and second call ─────────────────────────
    # finalize_scores_setup creates an OptimSubWorkspace and runs L-BFGS.
    # On try_enzyme: Optim v1.11.0 has a ManifoldObjective.value_gradient! bug
    # (returns scalar instead of (f, g)); this section is skipped if that fires.
    tee(io, "─── 3. finalize_scores! (1st and 2nd call) ──────────────────────\n")
    try
        GC.gc()
        t3a = @elapsed SSOF.finalize_scores!(mws)
        GC.gc()
        t3b = @elapsed SSOF.finalize_scores!(mws)
        tee(io, @sprintf("  1st call : %7.2f s\n", t3a))
        tee(io, @sprintf("  2nd call : %7.2f s\n", t3b))
    catch e
        tee(io, "  SKIPPED — $(typeof(e)): $(sprint(showerror, e))\n")
        tee(io, "  (Known Optim v1.11.0 ManifoldObjective bug; upgrade to v2+ to enable)\n")
    end
    tee(io, "\n")

    tee(io, "Results also written to: $outfile\n")
end
