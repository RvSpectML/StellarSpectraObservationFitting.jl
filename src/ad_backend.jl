# Backend seam: isolates the AD engine from the optimizers.
# Two functions form the full contract; implementing both is sufficient to add a backend.
#
# MooncakeBackend is available as a weak dependency via ext/SSOFMooncakeExt.jl.
# It is loaded automatically when the user also loads Mooncake.
#
# Remaining work (future PRs):
# - Enzyme.@import_rrule blocked by japi3 CallingConventionMismatchError (#2707);
#   _eval_lm_inner uses @from_rrule for Mooncake only — Enzyme traces through correctly.

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
Enzyme.EnzymeRules.inactive_type(::Type{<:Dict{Symbol, <:Real}}) = true

# Data subtypes (GenericData, LSFData, GenericDatum) hold observational flux,
# variance, and wavelength arrays that are never optimized parameters.
# inactive_type prevents "constant memory stored to differentiable variable"
# LLVM errors that arise when Enzyme fails to propagate Const through field
# accesses on a Data value captured in a Const closure.
Enzyme.EnzymeRules.inactive_type(::Type{<:Data}) = true

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
    # set_runtime_activity is required here for generic callables because Enzyme's
    # static activity analysis may not be able to prove activity through heterogeneous
    # containers. This path is hit by estimate_σ_curvature_helper and any direct
    # prepare_gradient callers that do not use FlatLoss. opt_funcs uses FlatLoss
    # (EnzymeFlatLossCache) which avoids this flag entirely.
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Duplicated(l, c.∂l),
        Enzyme.Active,
        Enzyme.Duplicated(θ, c.∂θ),
    )
    return val, c.∂θ
end

# ── FlatLoss: typed-nested path for opt_funcs ────────────────────────────────
#
# opt_funcs passes `f = loss ∘ unflatten` where unflatten is ParameterHandling's
# Vector_from_vec closure that returns Vector{Any}.  Differentiating f(θ) directly
# requires set_runtime_activity because Enzyme can't statically prove activity
# through Vector{Any} elements.
#
# FlatLoss instead differentiates loss(nested) w.r.t. nested directly (typed arrays,
# no Vector{Any}), then flattens the gradient. This reduces the number of runtime
# activity checks Enzyme must perform.  set_runtime_activity is still required because
# spectra_interp internally broadcasts Const SIH fields against Active RV arrays
# (sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)'), which Enzyme cannot prove
# statically.  The gain vs. the original path is fewer checks overall.

struct FlatLoss{L,U}
    loss::L
    unflatten::U
end
(f::FlatLoss)(x::AbstractVector) = f.loss(f.unflatten(x))

struct EnzymeFlatLossCache{∂N, T<:AbstractVector{<:Real}}
    ∂nested::∂N
    ∂θ::T
end

function prepare_gradient(::EnzymeBackend, f::FlatLoss, θ::AbstractVector{<:Real})
    nested = f.unflatten(θ)
    ∂nested = Enzyme.make_zero(nested)
    ∂θ = zero(θ)
    return EnzymeFlatLossCache(∂nested, ∂θ)
end

function value_and_gradient!(c::EnzymeFlatLossCache, f::FlatLoss, θ::AbstractVector{<:Real})
    nested = f.unflatten(θ)
    Enzyme.make_zero!(c.∂nested)
    _, val = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(f.loss),
        Enzyme.Active,
        Enzyme.Duplicated(nested, c.∂nested),
    )
    flat_grad, _ = ParameterHandling.flatten(c.∂nested)
    c.∂θ .= flat_grad
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

# Native EnzymeRules for spectra_interp (sparse-matrix signatures only).
# Mooncake uses ChainRulesCore rrules via @from_rrule.
# Math mirrors the ChainRulesCore rrule bodies in model_functions.jl.
# Two signatures are covered:
#   (1) column-wise vector-of-sparse (telluric path, om.t2o)
#   (2) single sparse matrix (LSF path, d.lsf)
# The stellar path spectra_interp(flux, rvs, SIH) is handled by Enzyme's native AD
# via set_runtime_activity (StellarInterpolationHelper declared inactive above).
# A custom SIH rule was tried but caused a 5× Adam regression because the Julia-level
# scatter-add in reverse() replaced cheaper LLVM-level AD.

# For Duplicated (array) return types, `dret` in `reverse` is a Type annotation,
# not an instance. The upstream gradient lives in the shadow, which is stored on
# the tape (same reference) so the reverse rule can read and zero it.

# (1) column-wise vector-of-sparse path
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

# (2) single sparse matrix path
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

# _rv_shift: element-wise sum of an active RV vector and a Const barycentric offset.
# The plain `rv .+ bary_rvs` would create Broadcasted{Tuple{Active,Const}}, which
# Enzyme cannot analyze statically (mixed-activity homogeneous tuple).  This named
# wrapper with an explicit Enzyme rule bypasses the Broadcasted intermediary.
_rv_shift(rv::AbstractVector{<:Real}, bary_rvs::AbstractVector{<:Real}) = rv .+ bary_rvs

function Enzyme.EnzymeRules.augmented_primal(
    ::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(_rv_shift)},
    ::Type,
    rv::Enzyme.Annotation{<:AbstractVector{<:Real}},
    bary_rvs::Enzyme.Annotation{<:AbstractVector{<:Real}},
)
    val = rv.val .+ bary_rvs.val
    shadow = zero(val)
    return Enzyme.EnzymeRules.AugmentedReturn(val, shadow, shadow)
end

function Enzyme.EnzymeRules.reverse(
    ::Enzyme.EnzymeRules.RevConfig,
    ::Enzyme.Const{typeof(_rv_shift)},
    ::Type,
    tape,
    rv::Enzyme.Annotation{<:AbstractVector{<:Real}},
    bary_rvs::Enzyme.Annotation{<:AbstractVector{<:Real}},
)
    if tape !== nothing
        !(rv isa Enzyme.Const) && (rv.dval .+= tape)
        !(bary_rvs isa Enzyme.Const) && (bary_rvs.dval .+= tape)
        tape .= 0
    end
    return (nothing, nothing)
end
