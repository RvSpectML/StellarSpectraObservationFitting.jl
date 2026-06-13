# Backend seam: isolates the AD engine from the optimizers.
# Two functions form the full contract; implementing both is sufficient to add a backend.
#
# Future EnzymeBackend requirements (out of scope for nabla-removal PR):
# - De-aliasing: workspace arrays alias θ entries (vec(lm) returns the model's own arrays,
#   so e.g. θ[2][3] === om.star.lm.μ), and loss closures capture om, o, d. Enzyme requires
#   declaring captured data Const or Duplicated; a Const closure holding arrays that alias
#   differentiated arguments is undefined behavior (typically "Constant memory is stored to
#   a differentiable variable" errors). Supporting Enzyme means either annotating the closure
#   Duplicated with a full shadow of captured structures, or restructuring losses so no
#   differentiated array is reachable through captured state.
# - DPCA mutation hoist (nabla-removal-plan.md Phase 2 step 6).
# - Enzyme.@import_rrule for the three ChainRulesCore rrules.
# - EnzymeBackend() added to backends_to_test in the Phase 3 gradient cross-check testset.
# - Package extension (ext/ + weakdep) so Mooncake users don't pay Enzyme's compile cost.

abstract type ADBackend end
struct MooncakeBackend <: ADBackend end

"""
    prepare_gradient(b::ADBackend, l, θ)

Build an opaque, backend-specific gradient cache for loss `l` at parameters `θ`.
`θ` is either a flat `Vector{Float64}` (Optim and error-estimation paths) or the
nested `Vector{<:AbstractArray}` used by the Adam path (entries may be SubArrays or
vectors of arrays).
"""
function prepare_gradient end

"""
    value_and_gradient!(cache, l, θ) -> (val::Real, ∂θ)

Return `(val, ∂θ)` where `∂θ` mirrors `θ` as plain nested Float64 arrays.
Mutates the cache in-place; not thread-safe.
"""
function value_and_gradient! end

# ── Mooncake implementation ───────────────────────────────────────────────────

import Mooncake

# TwicePrecision is immutable (Julia's internal double-double for range() steps).
# No copy() method exists for it, but Mooncake needs one when traversing OrderModel.
Base.copy(x::Base.TwicePrecision) = x

# Import the three ChainRulesCore rrules into Mooncake.
# Concrete-type registrations are most efficient when the static call-site types are
# concrete. Abstract-type registrations serve as fallbacks for call sites where struct
# field type declarations (e.g. t2o::AbstractVector{<:SparseMatrixCSC} in OrderModelWobble)
# prevent Mooncake from inferring the concrete type. Julia's method dispatch picks the
# most-specific matching rule, so both registrations coexist without ambiguity.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), Matrix{Float64}, Vector{Float64}, StellarInterpolationHelper}
# Stellar path: om.star.lm is declared LinearModel (abstract) in Submodel, so _eval_lm_vec
# returns an abstract-typed matrix at the call site. Register an abstract fallback.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), AbstractMatrix{Float64}, AbstractVector{<:Real}, StellarInterpolationHelper}
# Telluric path: om.t2o is declared AbstractVector{<:SparseMatrixCSC} in OrderModelWobble,
# so the static type at the call site is abstract. Register for both the abstract case
# and the concrete case (the concrete registration takes precedence when types are known).
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), Matrix{Float64}, Vector{SparseMatrixCSC{Float64,Int64}}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), AbstractMatrix{Float64}, AbstractVector{<:SparseMatrixCSC}}
# LSF path: d.lsf may be a single SparseMatrixCSC (same LSF for all observations).
# With the L type parameter on LSFData, d.lsf gets a concrete static type here.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(spectra_interp), AbstractMatrix{Float64}, SparseMatrixCSC{Float64,Int64}}
# gp_ℓ_precalc registration is deferred to prior_gp_functions.jl (defined there)
# _eval_lm_inner: covers the M*s+μ and exp(M*s).*μ paths called by _eval_lm_vec.
# Concrete Matrix{Float64} registrations only — the Adam path owns concrete arrays.
# The Optim path passes SubArrays (ParameterHandling.unflatten returns views), and
# Mooncake's SubArray tangent is an FData struct, not a plain Array; an AbstractMatrix
# fallback would misfire there and produce a tangent type mismatch. Mooncake falls
# through to generic tracing for SubArray inputs, which is correct but unoptimized.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Val{false}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Val{true}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Val{false}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Val{true}}
# TemplateModel path: _eval_lm(μ, n) = μ * ones(n)'. Same constraint applies.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_eval_lm), Vector{Float64}, Int}

# Recursive helper: Mooncake tangents for SubArrays may not be plain Arrays.
# The Adam θ can contain SubArrays (from downsize_view / vec(lm) on views);
# this helper ensures callers always receive plain nested Float64 arrays.
#
# For Vector{Vector} θ (TotalWorkspace), Mooncake returns tangents with
# Any-typed containers at multiple levels (Vector{Any} containing Vector{Any}
# containing float arrays). The Any overloads handle these recursively and
# retype to Vector{AbstractArray} so iterate!/first_iterate! dispatch works.
tangent_to_arrays(x::Array{<:Real}) = x                           # leaf: plain float array
tangent_to_arrays(x::AbstractArray{<:Real}) = collect(x)          # leaf: SubArray → plain Array
tangent_to_arrays(x::Tuple) = map(tangent_to_arrays, x)           # Tuple tangent (typed θ path)
tangent_to_arrays(x::AbstractVector{<:AbstractArray}) =            # typed container
    AbstractArray[tangent_to_arrays(xi) for xi in x]
tangent_to_arrays(x::AbstractArray{Any}) =                         # Any-typed container (Mooncake erasure)
    AbstractArray[tangent_to_arrays(xi) for xi in x]

struct MooncakeCache{R}
    rule::R
end

function prepare_gradient(::MooncakeBackend, l, θ; kwargs...)
    # Mooncake doesn't need aliasing info; kwargs (om, build_θ, build_l, o, d)
    # passed by Enzyme's nested-θ call sites are silently ignored here.
    rule = Mooncake.build_rrule(l, θ)
    return MooncakeCache(rule)
end

function value_and_gradient!(c::MooncakeCache, l, θ)
    val, (_, ∂θ) = Mooncake.value_and_gradient!!(c.rule, l, θ)
    return val, tangent_to_arrays(∂θ)
end

# ── Enzyme implementation ─────────────────────────────────────────────────────
#
# Flat-vector path (Optim + error_estimation) and nested-tuple path (Adam) both
# implemented. The nested path requires the caller to pass `om`, `build_θ`,
# `build_l`, `o`, `d` as kwargs to prepare_gradient so the cache can build
# shadow copies whose aliasing structure matches the primal.

import Enzyme

struct EnzymeBackend <: ADBackend end

# Cache for the flat-vector path. `∂l` is the shadow closure (preserves any
# alias-into-captured-state structure via Enzyme.make_zero's IdDict tracking).
# `∂θ` is pre-allocated; zeroed in-place each call.
struct EnzymeFlatCache{L, T<:AbstractVector{<:Real}}
    ∂l::L
    ∂θ::T
end

function prepare_gradient(::EnzymeBackend, l, θ::AbstractVector{<:Real})
    ∂l = Enzyme.make_zero(l)
    ∂θ = zero(θ)
    return EnzymeFlatCache(∂l, ∂θ)
end

function value_and_gradient!(c::EnzymeFlatCache, l, θ::AbstractVector{<:Real})
    # remake_zero! (not make_zero!) is required when the captured closure
    # contains differentiable Float64 values in immutable struct positions —
    # e.g. StellarInterpolationHelper's AbstractMatrix{Float64} field. The
    # shadow ∂l was built via make_zero, so those positions are already zero;
    # remake_zero! preserves that without re-checking. make_zero! would error.
    Enzyme.remake_zero!(c.∂l)
    fill!(c.∂θ, 0)
    # set_runtime_activity is required for the Optim/error_estimation path,
    # where `l = loss ∘ unflatten` and unflatten is ParameterHandling's
    # Vector_from_vec closure. Enzyme's static activity analysis cannot prove
    # the array-of-views inside Vector_from_vec is fully active, so it errors
    # with EnzymeRuntimeActivityError without this flag. The runtime check
    # has a small per-call cost but is unconditionally correct. If Phase 7
    # profiling shows it dominating, the alternative is to rewrite the
    # affected losses to not flow constants into differentiable storage.
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Duplicated(l, c.∂l),
        Enzyme.Active,
        Enzyme.Duplicated(θ, c.∂θ),
    )
    return val, c.∂θ
end

# Cache for the nested-tuple path used by the Adam optimizer. The Adam path's
# θ entries alias arrays inside a captured OrderModel — e.g. when the caller
# constructs `θ = (om.tel.lm.s, om.star.lm.s, om.rv)`, the array `θ[1]` is
# the same object as `om.tel.lm.s`. To get correct gradients under Enzyme we
# need the shadow `∂θ` to alias `∂l`'s captured `∂om` arrays in exactly the
# same way; otherwise Enzyme's per-pointer tangent tracking double-counts or
# misses gradient contributions.
#
# Construction recipe: deepcopy-zero `om` into `∂om` (Enzyme.make_zero
# preserves shared-structure identity via IdDict), then call the caller's
# `build_l(∂om, o, d)` and `build_θ(∂om)` functions to reconstruct the
# shadow closure and shadow tuple. Because both `build_l` and `build_θ`
# read array fields from the same `∂om`, the resulting `∂l`-captures and
# `∂θ`-leaves naturally share identity within `∂om`.
#
# `remake_zero!(c.∂om)` at the start of each gradient call zeros every
# Float64 array reachable from `∂om`. Since `∂l`'s captures and `∂θ`'s
# leaves all alias into `∂om`, that single in-place zeroing prepares both.
struct EnzymeNestedCache{∂L, ∂T, ∂OM}
    ∂l::∂L
    ∂θ::∂T
    ∂om::∂OM  # held alive so ∂l and ∂θ aliases stay valid
end

function prepare_gradient(::EnzymeBackend, l, θ::Tuple; om, build_θ, build_l, o, d)
    ∂om = Enzyme.make_zero(om)
    ∂l  = build_l(∂om, o, d)
    ∂θ  = build_θ(∂om)
    return EnzymeNestedCache(∂l, ∂θ, ∂om)
end

function value_and_gradient!(c::EnzymeNestedCache, l, θ::Tuple)
    Enzyme.remake_zero!(c.∂om)
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Duplicated(l, c.∂l),
        Enzyme.Active,
        Enzyme.Duplicated(θ, c.∂θ),
    )
    return val, c.∂θ
end

# ChainRulesCore rrule import: BLOCKED on Julia 1.12.
#
# Enzyme.@import_rrule's generated reverse function uses the `japi3` calling
# convention which Enzyme 0.13's LLVM bridge does not yet support on Julia
# 1.12 (CallingConventionMismatchError; tracking issue EnzymeAD/Enzyme.jl#2707).
# Until that is resolved, the EnzymeBackend differentiates through
# spectra_interp / gp_ℓ_precalc / _eval_lm_inner / _eval_lm directly via
# native Enzyme AD (correct but slower than the imported rrules would be).
#
# Phase 7 perf path forward (in priority order):
#   1. Reassess after upgrading Enzyme (issue #2707).
#   2. Write native Enzyme.EnzymeRules.augmented_primal / reverse for the four
#      hot rules (spectra_interp ×2, gp_ℓ_precalc, _eval_lm_inner). The math
#      is in the ChainRulesCore rrule bodies; only the activity-shadow
#      bookkeeping needs translation.
#   3. Downgrade benchmarking to Julia 1.11, where @import_rrule is reported
#      working. Not a fix — just a baseline.
