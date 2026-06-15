#!/usr/bin/env julia
# test_finalize_scores_regression.jl
#
# Regression test: compare finalize_scores! output between master (Nabla) and
# try_enzyme (Enzyme) by loading precomputed master results and checking that
# Enzyme produces numerically close RVs and a lower or equal final loss.
#
# Run from the repo root on try_enzyme branch:
#   julia test_finalize_scores_regression.jl
#
# Notes:
# - om.rv for OrderModelWobble is stored in m/s (differential stellar RVs after
#   barycentric correction); no unit conversion needed.
# - Master uses data.jld2.bak because data.jld2 was saved with try_enzyme's
#   4-parameter LSFData{T,AM,M,L} struct, which master's 3-parameter struct
#   cannot reconstruct via JLD2.

const JULIA = "/home/eford/.julia/juliaup/julia-1.11.2+0.x64.linux.gnu/bin/julia"

using Pkg

const REPO   = dirname(abspath(@__FILE__))
const MASTER = "/tmp/ssof-master"

# ── Run Enzyme (try_enzyme) finalize_scores! ──────────────────────────────────
env = mktempdir(; cleanup=true)
Pkg.activate(env)
Pkg.develop(PackageSpec(path=REPO); io=devnull)
Pkg.add(["JLD2", "Statistics", "Printf"]; io=devnull)

import StellarSpectraObservationFitting as SSOF
using JLD2, Statistics, Printf

data_dir = joinpath(REPO, "examples", "data")
@load joinpath(data_dir, "results.jld2") model
@load joinpath(data_dir, "data.jld2") data

mws_en = SSOF.ModelWorkspace(model, data)
loss_before_en = SSOF._loss(mws_en)
SSOF.finalize_scores!(mws_en)
loss_after_en = SSOF._loss(mws_en)
rv_after_en   = copy(mws_en.om.rv)

println("=== Enzyme (try_enzyme) ===")
@printf("  loss before : %.6f\n", loss_before_en)
@printf("  loss after  : %.6f\n", loss_after_en)
@printf("  loss Δ      : %.6e\n", loss_after_en - loss_before_en)
@printf("  RV range    : [%.4f, %.4f] m/s\n", minimum(rv_after_en), maximum(rv_after_en))

# ── Load master (Nabla) results from a separate session ──────────────────────
master_rv_file = "/tmp/regression_master_rv.jld2"

let
    master_env = mktempdir(; cleanup=true)
    Pkg.activate(master_env)
    Pkg.develop(PackageSpec(path=MASTER); io=devnull)
    Pkg.add(["JLD2"]; io=devnull)

    master_code = """
    import StellarSpectraObservationFitting as SSOF
    using JLD2
    data_dir = "$(data_dir)"
    @load joinpath(data_dir, "results.jld2") model
    @load joinpath(data_dir, "data.jld2.bak") data  # .bak: master's 3-param LSFData
    mws = SSOF.ModelWorkspace(model, data)
    loss_before = SSOF._loss(mws)
    SSOF.finalize_scores!(mws)
    loss_after = SSOF._loss(mws)
    rv = copy(mws.om.rv)
    @save "$(master_rv_file)" rv loss_before loss_after
    println("Master loss before: ", loss_before)
    println("Master loss after:  ", loss_after)
    """
    master_script = joinpath(REPO, "_tmp_master_regression.jl")
    write(master_script, master_code)
    run(`$(JULIA) --project=$(master_env) $(master_script)`)
    rm(master_script)
end

Pkg.activate(env)
@load master_rv_file rv loss_before loss_after
rv_master      = rv
loss_before_ma = loss_before
loss_after_ma  = loss_after

println("\n=== Nabla (master) ===")
@printf("  loss before : %.6f\n", loss_before_ma)
@printf("  loss after  : %.6f\n", loss_after_ma)
@printf("  loss Δ      : %.6e\n", loss_after_ma - loss_before_ma)
@printf("  RV range    : [%.4f, %.4f] m/s\n", minimum(rv_master), maximum(rv_master))

# ── Compare ───────────────────────────────────────────────────────────────────
rv_diff = rv_after_en .- rv_master   # both already in m/s
println("\n=== Regression comparison ===")
@printf("  RV max |Δ|   : %.6f m/s\n", maximum(abs, rv_diff))
@printf("  RV median |Δ|: %.6f m/s\n", median(abs.(rv_diff)))
@printf("  loss_after Enzyme : %.10f\n", loss_after_en)
@printf("  loss_after Nabla  : %.10f\n", loss_after_ma)
@printf("  loss Δ (En-Na)    : %.6e\n", loss_after_en - loss_after_ma)

max_rv_diff = maximum(abs, rv_diff)
if max_rv_diff < 1.0
    println("\n✓ PASS  max RV deviation = $(round(max_rv_diff; digits=6)) m/s  (threshold 1.0 m/s)")
else
    println("\n✗ FAIL  max RV deviation = $(round(max_rv_diff; digits=6)) m/s  (threshold 1.0 m/s)")
end
if loss_after_en <= loss_after_ma * 1.01
    @printf("✓ PASS  Enzyme final loss ≤ 1%% above Nabla  (Δ = %.3e)\n", loss_after_en - loss_after_ma)
else
    @printf("✗ FAIL  Enzyme final loss significantly higher than Nabla  (Δ = %.3e)\n", loss_after_en - loss_after_ma)
end
