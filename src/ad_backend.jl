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

function prepare_gradient(::MooncakeBackend, l, θ)
    rule = Mooncake.build_rrule(l, θ)
    return MooncakeCache(rule)
end

function value_and_gradient!(c::MooncakeCache, l, θ)
    val, (_, ∂θ) = Mooncake.value_and_gradient!!(c.rule, l, θ)
    return val, tangent_to_arrays(∂θ)
end
