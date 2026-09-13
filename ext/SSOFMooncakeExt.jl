module SSOFMooncakeExt

import Mooncake
import StellarSpectraObservationFitting as SSOF
using SparseArrays: SparseMatrixCSC

# TwicePrecision is immutable (Julia's internal double-double for range() steps).
# No copy() method exists for it, but Mooncake needs one when traversing OrderModel.
Base.copy(x::Base.TwicePrecision) = x

# ── ChainRulesCore rrule → Mooncake registrations ────────────────────────────
#
# Concrete-type registrations are most efficient when the static call-site types
# are concrete.  Abstract-type registrations serve as fallbacks for call sites
# where struct field type declarations (e.g. t2o::AbstractVector{<:SparseMatrixCSC}
# in OrderModelWobble) prevent Mooncake from inferring the concrete type.
# Julia's method dispatch picks the most-specific matching rule, so both
# registrations coexist without ambiguity.

# spectra_interp — stellar (StellarInterpolationHelper) path
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.spectra_interp), Matrix{Float64}, Vector{Float64}, SSOF.StellarInterpolationHelper}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.spectra_interp), AbstractMatrix{Float64}, AbstractVector{<:Real}, SSOF.StellarInterpolationHelper}

# spectra_interp — telluric (vector-of-sparse) path
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.spectra_interp), Matrix{Float64}, Vector{SparseMatrixCSC{Float64,Int64}}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.spectra_interp), AbstractMatrix{Float64}, AbstractVector{<:SparseMatrixCSC}}

# spectra_interp — LSF (single sparse matrix) path
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.spectra_interp), AbstractMatrix{Float64}, SparseMatrixCSC{Float64,Int64}}

# _eval_lm_inner — M*s+μ and exp(M*s).*μ paths called by _eval_lm_vec.
# Concrete Matrix{Float64} registrations only: the Optim path (via FlatLoss)
# differentiates loss(nested) in nested space; ParameterHandling.flatten returns
# owned Matrix/Vector elements (not SubArrays).  AbstractMatrix fallback is omitted
# because Mooncake's tangent type for abstract containers may mismatch; concrete
# registrations are sufficient for the arrays produced by unflatten.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF._eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Val{false}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF._eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Val{true}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF._eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Val{false}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF._eval_lm_inner), Matrix{Float64}, Matrix{Float64}, Val{true}}

# _eval_lm — TemplateModel path: _eval_lm(μ, n) = μ * ones(n)'
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF._eval_lm), Vector{Float64}, Int}

# gp_ℓ_precalc — concrete-type and abstract-type fallback registrations.
# The rrule is defined in prior_gp_functions.jl via ChainRulesCore.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.gp_ℓ_precalc), Matrix{Float64}, Vector{Float64}, SSOF.SteadyStateGP{Float64}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(SSOF.gp_ℓ_precalc), AbstractArray, AbstractVector, SSOF.SteadyStateGP}

# ── Tangent helpers ───────────────────────────────────────────────────────────
#
# Mooncake tangents for SubArrays may not be plain Arrays.
# The Adam θ can contain SubArrays (from downsize_view / vec(lm) on views);
# this helper ensures callers always receive plain nested Float64 arrays.
#
# For Vector{Vector} θ (TotalWorkspace), Mooncake returns tangents with
# Any-typed containers at multiple levels (Vector{Any} containing Vector{Any}
# containing float arrays).  The Any overloads handle these recursively and
# retype to Vector{AbstractArray} so iterate!/first_iterate! dispatch works.
tangent_to_arrays(x::Array{<:Real}) = x
tangent_to_arrays(x::AbstractArray{<:Real}) = collect(x)
tangent_to_arrays(x::Tuple) = map(tangent_to_arrays, x)
tangent_to_arrays(x::AbstractVector{<:AbstractArray}) =
    AbstractArray[tangent_to_arrays(xi) for xi in x]
tangent_to_arrays(x::AbstractArray{Any}) =
    AbstractArray[tangent_to_arrays(xi) for xi in x]

# ── MooncakeBackend implementation ────────────────────────────────────────────

struct MooncakeCache{R}
    rule::R
end

function SSOF.prepare_gradient(::SSOF.MooncakeBackend, l, θ; kwargs...)
    # Mooncake doesn't need aliasing info; kwargs (om, build_θ, build_l, o, d)
    # passed by Enzyme's nested-θ call sites are silently ignored here.
    rule = Mooncake.build_rrule(l, θ)
    return MooncakeCache(rule)
end

function SSOF.value_and_gradient!(c::MooncakeCache, l, θ)
    val, (_, ∂θ) = Mooncake.value_and_gradient!!(c.rule, l, θ)
    return val, tangent_to_arrays(∂θ)
end

end # module SSOFMooncakeExt
