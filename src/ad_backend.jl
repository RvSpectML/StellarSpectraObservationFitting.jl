# Backend seam: isolates the AD engine from the optimizers.
# Two functions form the full contract; implementing both is sufficient to add a backend.
#
# Remaining work (future PRs):
# - DPCA mutation hoist: _eval_lm_vec in DPCA uses mutation; needs Const annotation or refactor.
# - Enzyme.@import_rrule blocked by japi3 CallingConventionMismatchError (#2707);
#   _eval_lm_inner uses @from_rrule for Mooncake only — Enzyme traces through correctly.
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

# StellarInterpolationHelper holds precomputed interpolation indices and weights
# derived from the fixed observational and model wavelength grids.  It is never
# a function of the optimized parameters, so its tangent is always zero —
# declare it inactive so Enzyme skips MixedDuplicated construction for it.
Enzyme.EnzymeRules.inactive_type(::Type{<:StellarInterpolationHelper}) = true

# Regularization dicts (reg_tel, reg_star) hold constant Float64 coefficients
# that are never a function of the optimized parameters. Declaring Dict as
# inactive prevents Enzyme from attempting to construct shadows for dict
# internals when activity analysis fails to propagate Const through field access.
Enzyme.EnzymeRules.inactive_type(::Type{<:AbstractDict}) = true

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

# Recursive helpers for the nested-Tuple θ used by the Adam nested path.
# These maintain a non-aliased pre-allocated copy buffer and gradient buffer.
_enzyme_copy_nested(θ::Tuple) = map(_enzyme_copy_nested, θ)
_enzyme_copy_nested(a::AbstractArray) = copy(a)
_enzyme_zero_like(θ::Tuple) = map(_enzyme_zero_like, θ)
_enzyme_zero_like(a::AbstractArray) = zero(a)
function _enzyme_copy_nested!(dst::Tuple, src::Tuple)
    for (d, s) in zip(dst, src)
        _enzyme_copy_nested!(d, s)
    end
end
_enzyme_copy_nested!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
function _enzyme_zero_nested!(t::Tuple)
    for x in t
        _enzyme_zero_nested!(x)
    end
end
_enzyme_zero_nested!(a::AbstractArray) = fill!(a, 0)

# Cache for the nested-tuple path used by the Adam optimizer.
#
# The Adam path's θ entries alias arrays inside the captured OrderModel
# (e.g. θ[1][1] === l.om.tel.lm.M). Naively using both Duplicated(l, ∂l)
# and Duplicated(θ, ∂θ) causes Enzyme's runtime activity analysis to see
# the same primal pointer covered by two Duplicated annotations and demote
# one to Const — silently zeroing gradients for s, μ, and rv.
#
# Fix: declare the loss closure Const (no shadow). This is valid because
# l_total(total) accesses all differentiable parameters through its argument
# `total` (i.e., total[1], total[2], total[3]) and never through om.*.lm.*
# directly. The captured om, o, d provide only non-differentiable data
# (wavelength grids, interpolation helpers, regularization coefficients).
# θ_copy is a fresh non-aliased copy allocated once; it is synced from the
# live θ each call, so Enzyme sees no shared pointers between Const(l) and
# Duplicated(θ_copy, ∂θ).
struct EnzymeNestedCache{T, ∂T}
    θ_copy::T   # pre-allocated non-aliased copy of θ
    ∂θ::∂T      # gradient output
end

function prepare_gradient(::EnzymeBackend, l, θ::Tuple; kwargs...)
    # kwargs (om, build_θ, build_l, o, d) accepted for API compatibility; not used.
    θ_copy = _enzyme_copy_nested(θ)
    ∂θ = _enzyme_zero_like(θ)
    return EnzymeNestedCache(θ_copy, ∂θ)
end

function value_and_gradient!(c::EnzymeNestedCache, l, θ::Tuple)
    _enzyme_copy_nested!(c.θ_copy, θ)
    _enzyme_zero_nested!(c.∂θ)
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(l),
        Enzyme.Active,
        Enzyme.Duplicated(c.θ_copy, c.∂θ),
    )
    return val, c.∂θ
end

# Native EnzymeRules for spectra_interp (sparse-matrix interpolation).
# Mooncake uses ChainRulesCore rrules via @from_rrule; Enzyme traces through
# SparseMatrixCSC operations and produces wrong gradients without these rules.
# Math mirrors the ChainRulesCore rrule bodies in model_functions.jl.
# These cover the two signatures actually used in the χ² loss:
#   (1) column-wise vector-of-sparse (telluric path, om.t2o)
#   (2) single sparse matrix (LSF path, d.lsf)
# The stellar path spectra_interp(flux, rvs, SIH) is handled correctly by
# Enzyme's native AD (StellarInterpolationHelper declared inactive above).

# For Duplicated (array) return types, `dret` in `reverse` is a Type annotation,
# not an instance. The upstream gradient lives in the shadow, which is stored on
# the tape (same reference) so the reverse rule can read and zero it.

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(spectra_interp)},
    ::Type,
    model::Enzyme.Annotation,
    interp_helper::Enzyme.Annotation{<:AbstractVector{<:SparseMatrixCSC}},
)
    primal_val = spectra_interp(model.val, interp_helper.val)
    shadow = zero(primal_val)
    return Enzyme.EnzymeRules.AugmentedReturn(primal_val, shadow, shadow)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(spectra_interp)},
    ::Type,
    tape,
    model::Enzyme.Annotation,
    interp_helper::Enzyme.Annotation{<:AbstractVector{<:SparseMatrixCSC}},
)
    if !(model isa Enzyme.Const) && tape !== nothing
        for i in axes(model.val, 2)
            model.dval[:, i] .+= interp_helper.val[i]' * tape[:, i]
        end
        tape .= 0
    end
    return (nothing, nothing)
end

function Enzyme.EnzymeRules.augmented_primal(
    config::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(spectra_interp)},
    ::Type,
    model::Enzyme.Annotation,
    interp_helper::Enzyme.Annotation{<:SparseMatrixCSC},
)
    primal_val = spectra_interp(model.val, interp_helper.val)
    shadow = zero(primal_val)
    return Enzyme.EnzymeRules.AugmentedReturn(primal_val, shadow, shadow)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(spectra_interp)},
    ::Type,
    tape,
    model::Enzyme.Annotation,
    interp_helper::Enzyme.Annotation{<:SparseMatrixCSC},
)
    if !(model isa Enzyme.Const) && tape !== nothing
        model.dval .+= interp_helper.val' * tape
        tape .= 0
    end
    return (nothing, nothing)
end
