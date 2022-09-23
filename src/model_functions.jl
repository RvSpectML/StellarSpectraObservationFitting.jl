using AbstractGPs
using LinearAlgebra
using KernelFunctions
using TemporalGPs
import AbstractGPs
using Distributions
import Base.copy
import Base.vec
import Base.getindex
import Base.eachindex
import Base.setindex!
using SparseArrays
using SpecialFunctions
using StaticArrays
using Nabla
import StatsBase: winsor
using Base.Threads
using ThreadsX

abstract type OrderModel end
abstract type Output end
abstract type Data end

_current_matrix_modifier = SparseMatrixCSC

function D_to_rv(D)
	x = exp.(2 .* D)
	return light_speed_nu .* ((1 .- x) ./ (1 .+ x))
end

rv_to_D(v) = (log1p.((-v ./ light_speed_nu)) - log1p.((v ./ light_speed_nu))) ./ 2
# rv_to_D(v) = log.((1 .- v ./ light_speed_nu) ./ (1 .+ v ./ light_speed_nu)) ./ 2
function _lower_inds!(lower_inds::AbstractMatrix, lower_inds_adj::AbstractMatrix, model_log_λ::AbstractVector{<:Real}, rvs, log_λ_obs::AbstractMatrix)
	n_obs = length(rvs)
	len = size(log_λ_obs, 1)
	@assert size(lower_inds) == size(lower_inds_adj) == (len, n_obs)
	log_λ_holder = Array{Float64}(undef, len)
	len_model = length(model_log_λ)
	for i in 1:n_obs
		log_λ_holder[:] = view(log_λ_obs, :, i) .+ rv_to_D(rvs[i])
		lower_inds[:, i] .= searchsortednearest(model_log_λ, log_λ_holder; lower=true)
		for j in 1:len
			if lower_inds[j, i] >= len_model
				lower_inds[j, i] = len_model - 1
			elseif lower_inds[j, i] < 1
				lower_inds[j, i] = 1
			end
		end
		lower_inds_adj[:, i] .= ((i - 1) * len_model) .+ view(lower_inds, :, i)
	end
	return lower_inds, lower_inds_adj
end
function _lower_inds(model_log_λ::AbstractVector{<:Real}, rvs, log_λ_obs::AbstractMatrix)
	n_obs = length(rvs)
	len = size(log_λ_obs, 1)
	lower_inds = Array{Int64}(undef, len, n_obs)
	lower_inds_adj = Array{Int64}(undef, len, n_obs)
	return _lower_inds!(lower_inds, lower_inds_adj, model_log_λ, rvs, log_λ_obs)
end

struct StellarInterpolationHelper{T1<:Real, T2<:Int}
    log_λ_obs_m_model_log_λ_lo::AbstractMatrix{T1}
	model_log_λ_step::T1
	lower_inds::AbstractMatrix{T2}
	lower_inds_p1::AbstractMatrix{T2}
	function StellarInterpolationHelper(
		log_λ_obs_m_model_log_λ_lo::AbstractMatrix{T1},
		model_log_λ_step::T1,
		lower_inds::AbstractMatrix{T2},
		lower_inds_p1::AbstractMatrix{T2}) where {T1<:Real, T2<:Int}
		# @assert some issorted thing?
		return new{T1, T2}(log_λ_obs_m_model_log_λ_lo, model_log_λ_step, lower_inds, lower_inds_p1)
	end
end
function StellarInterpolationHelper(
	model_log_λ::StepRangeLen,
	rvs::AbstractVector{T},
	log_λ_obs::AbstractMatrix{T}) where {T<:Real}

	sih = StellarInterpolationHelper(Array{Float64}(undef, size(log_λ_obs)), model_log_λ.step.hi, Array{Int}(undef, size(log_λ_obs)), Array{Int}(undef, size(log_λ_obs)))
	return StellarInterpolationHelper!(sih, model_log_λ, rvs, log_λ_obs)
end
function (sih::StellarInterpolationHelper)(inds::AbstractVecOrMat, len_model::Int)
	lower_inds = copy(sih.lower_inds)
	for i in 1:length(inds)
		j = inds[i]
		lower_inds[:, j] .+= (i - j) * len_model
	end
	return StellarInterpolationHelper(
		view(sih.log_λ_obs_m_model_log_λ_lo, :, inds),
		sih.model_log_λ_step,
		lower_inds[:, inds],
		lower_inds[:, inds] .+ 1)
end
function StellarInterpolationHelper!(
	sih::StellarInterpolationHelper,
	model_log_λ::StepRangeLen,
	total_rvs::AbstractVector{T},
	log_λ_obs::AbstractMatrix{T}) where {T<:Real}

	@assert sih.model_log_λ_step == model_log_λ.step.hi
	lower_inds = Array{Int}(undef, size(sih.lower_inds, 1), size(sih.lower_inds, 2))
	_lower_inds!(lower_inds, sih.lower_inds, model_log_λ, total_rvs, log_λ_obs)

	sih.log_λ_obs_m_model_log_λ_lo .= log_λ_obs - (view(model_log_λ, lower_inds))
	sih.lower_inds_p1 .= sih.lower_inds .+ 1
	return sih
end

function spectra_interp(model_flux::AbstractVector, rv::Real, sih::StellarInterpolationHelper; sih_ind::Int=1)
	ratios = (view(sih.log_λ_obs_m_model_log_λ_lo, :, sih_ind) .+ rv_to_D(rv)) ./ sih.model_log_λ_step
	return (view(model_flux, view(sih.lower_inds, :, sih_ind)) .* (1 .- ratios)) + (view(model_flux, view(sih.lower_inds_p1, :, sih_ind)) .* ratios)
end
function spectra_interp(model_flux::AbstractMatrix, rvs::AbstractVector, sih::StellarInterpolationHelper)
	ratios = (sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)') ./ sih.model_log_λ_step
	# prop_bad = sum((ratios .> 1) + (ratios .< 0)) / length(ratios)
	# if prop_bad > 0.01; println("$(Int(round((100*prop_bad))))% of ratios are outside [0,1]. Consider running update_interpolation_locations!(::ModelWorkspace)")
	return (view(model_flux, sih.lower_inds).* (1 .- ratios)) + (view(model_flux, sih.lower_inds_p1) .* ratios)
end
function spectra_interp_nabla(model_flux, rvs, sih::StellarInterpolationHelper)
	ratios = (sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)') ./ sih.model_log_λ_step
	return (model_flux[sih.lower_inds] .* (1 .- ratios)) + (model_flux[sih.lower_inds_p1] .* ratios)
end
spectra_interp(model_flux, rvs, sih::StellarInterpolationHelper) =
	spectra_interp_nabla(model_flux, rvs, sih)
@explicit_intercepts spectra_interp Tuple{AbstractMatrix, AbstractVector, StellarInterpolationHelper} [true, false, false]
function Nabla.∇(::typeof(spectra_interp), ::Type{Arg{1}}, _, y, ȳ, model_flux, rvs, sih)
	ratios = (sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)') ./ sih.model_log_λ_step
	ȳnew = zeros(size(model_flux, 1), size(ȳ, 2))
	# samp is λ_obs x λ_model
	for k in 1:size(ȳ, 1)  # λ_obs
		for j in 1:size(ȳnew, 2)  # time
			λ_model_lo = sih.lower_inds[k, j] - size(ȳnew, 1)*(j-1)
			# for i in 1:size(ȳnew, 1)  # λ_model
			# for i in λ_model_lo:(λ_model_lo+1) # λ_model
			# ȳnew[i, j] += sampt[i, k] * ȳ[k, j]
			# ȳnew[i, j] += samp[k, i] * ȳ[k, j]
			# ȳnew[λ_model_lo, j] += samp[k, λ_model_lo] * ȳ[k, j]
			ȳnew[λ_model_lo, j] += (1 - ratios[k, j]) * ȳ[k, j]
			ȳnew[λ_model_lo+1, j] += ratios[k, j] * ȳ[k, j]
			# end
		end
	end
	return ȳnew
end

struct LSFData{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{<:Number}} <: Data
    flux::AM
    var::AM
    log_λ_obs::AM
	log_λ_obs_bounds::M
    log_λ_star::AM
	log_λ_star_bounds::M
	lsf::_current_matrix_modifier
	function LSFData(flux::AM, var::AM, log_λ_obs::AM, log_λ_star::AM, lsf::_current_matrix_modifier) where {T<:Real, AM<:AbstractMatrix{T}}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		@assert size(lsf, 1) == size(lsf, 2) == size(flux, 1)
		log_λ_obs_bounds = bounds_generator(log_λ_obs)
		log_λ_star_bounds = bounds_generator(log_λ_star)
		return new{T, AM, typeof(log_λ_obs_bounds)}(flux::AM, var::AM, log_λ_obs::AM, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds, lsf)
	end
end
LSFData(flux::AM, var::AM, log_λ_obs::AM, log_λ_star::AM, lsf::Nothing) where {T<:Real, AM<:AbstractMatrix{T}} =
	GenericData(flux, var, log_λ_obs, log_λ_star)
(d::LSFData)(inds::AbstractVecOrMat) =
	LSFData(view(d.flux, :, inds), view(d.var, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds), d.lsf)
Base.copy(d::LSFData) = LSFData(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star), copy(d.lsf))

struct GenericData{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{<:Number}} <: Data
    flux::AM
    var::AM
    log_λ_obs::AM
	log_λ_obs_bounds::M
    log_λ_star::AM
	log_λ_star_bounds::M
	function GenericData(flux::AM, var::AM, log_λ_obs::AM, log_λ_star::AM) where {T<:Number, AM<:AbstractMatrix{T}}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		log_λ_obs_bounds = bounds_generator(log_λ_obs)
		log_λ_star_bounds = bounds_generator(log_λ_star)
		return new{T, AM, typeof(log_λ_obs_bounds)}(flux, var, log_λ_obs, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds)
	end
end
(d::GenericData)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, :, inds), view(d.var, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds))
Base.copy(d::GenericData) = GenericData(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star))
GenericData(d::LSFData) = GenericData(d.flux, d.var, d.log_λ_obs, d.log_λ_star)
GenericData(d::GenericData) = d

struct GenericDatum{T<:Number, AV<:AbstractVector{T}, V<:Vector{<:Number}} <: Data
    flux::AV
    var::AV
    log_λ_obs::AV
	log_λ_obs_bounds::V
    log_λ_star::AV
	log_λ_star_bounds::V
	function GenericDatum(flux::AV, var::AV, log_λ_obs::AV, log_λ_star::AV) where {T<:Number, AV<:AbstractVector{T}}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		log_λ_obs_bounds = bounds_generator(log_λ_obs)
		log_λ_star_bounds = bounds_generator(log_λ_star)
		return new{T, AV, typeof(log_λ_obs_bounds)}(flux, var, log_λ_obs, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds)
	end
end
(d::GenericDatum)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, inds), view(d.var, inds),
	view(d.log_λ_obs, inds), view(d.log_λ_star, inds))
Base.copy(d::GenericDatum) = GenericDatum(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star))
function GenericData(d::Vector{<:GenericDatum})
	len_obs = length(d[1].flux)
	n_obs = length(d)
	flux_obs = ones(len_obs, n_obs)
	var_obs = Array{Float64}(undef, len_obs, n_obs)
	log_λ_obs = Array{Float64}(undef, len_obs, n_obs)
	# log_λ_obs_bounds = Array{Float64}(undef, len_obs+1, n_obs)
	log_λ_star = Array{Float64}(undef, len_obs, n_obs)
	# log_λ_star_bounds = Array{Float64}(undef, len_obs+1, n_obs)
	for i in 1:n_obs # 13s
		flux_obs[:, i] .= d[i].flux
		var_obs[:, i] .= d[i].var
		log_λ_obs[:, i] .= d[i].log_λ_obs
		# log_λ_obs_bounds[:, i] .= d[i].log_λ_obs_bounds
		log_λ_star[:, i] .= d[i].log_λ_star
		# log_λ_star_bounds[:, i] .= d[i].log_λ_star_bounds
	end
	return GenericData(flux_obs, var_obs, log_λ_obs, log_λ_star)
end

function create_λ_template(log_λ_obs::AbstractMatrix; upscale::Real=1.)
    log_min_wav, log_max_wav = extrema(log_λ_obs)
	Δ_logλ_og = minimum(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # minimum pixel sep
	# Δ_logλ_og = minimum(view(log_λ_obs, size(log_λ_obs, 1), :) .- view(log_λ_obs, 1, :)) / size(log_λ_obs, 1)  # minimum avg pixel sep
	# Δ_logλ_og = median(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # median pixel sep
	# Δ_logλ_og = maximum(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # maximum pixel sep
	Δ_logλ = Δ_logλ_og / upscale
    log_λ_template = (log_min_wav - 2 * Δ_logλ_og):Δ_logλ:(log_max_wav + 2 * Δ_logλ_og)
    λ_template = exp.(log_λ_template)
    return log_λ_template, λ_template
end

abstract type LinearModel end
_log_lm_default = false

# Full (includes mean) linear model
struct FullLinearModel{T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}, AV<:AbstractVector{T}}  <: LinearModel
	M::AM1
	s::AM2
	μ::AV
	log::Bool
	function FullLinearModel(M::AM1, s::AM2, μ::AV, log::Bool) where {T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}, AV<:AbstractVector{T}}
		@assert length(μ) == size(M, 1)
		@assert size(M, 2) == size(s, 1)
		return new{T, AM1, AM2, AV}(M, s, μ, log)
	end
end
FullLinearModel(M::AM1, s::AM2, μ::AV; log::Bool=_log_lm_default) where {T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}, AV<:AbstractVector{T}} =
	FullLinearModel(M, s, μ, log)
Base.copy(flm::FullLinearModel) = FullLinearModel(copy(flm.M), copy(flm.s), copy(flm.μ), flm.log)
LinearModel(flm::FullLinearModel, inds::AbstractVecOrMat) =
	FullLinearModel(flm.M, view(flm.s, :, inds), flm.μ, flm.log)

# Base (no mean) linear model
struct BaseLinearModel{T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}} <: LinearModel
	M::AM1
	s::AM2
	log::Bool
	function BaseLinearModel(M::AM1, s::AM2, log::Bool) where {T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}}
		@assert size(M, 2) == size(s, 1)
		return new{T, AM1, AM2}(M, s, log)
	end
end
BaseLinearModel(M::AM1, s::AM2; log::Bool=_log_lm_default) where {T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}} =
	BaseLinearModel(M, s, log)
Base.copy(blm::BaseLinearModel) = BaseLinearModel(copy(blm.M), copy(blm.s), blm.log)
LinearModel(blm::BaseLinearModel, inds::AbstractVecOrMat) =
	BaseLinearModel(blm.M, view(blm.s, :, inds), blm.log)

# Template (no bases) model
struct TemplateModel{T<:Number, AV<:AbstractVector{T}} <: LinearModel
	μ::AV
	n::Int
end
Base.copy(tlm::TemplateModel) = TemplateModel(copy(tlm.μ), tlm.n)
LinearModel(tm::TemplateModel, inds::AbstractVecOrMat) = TemplateModel(tm.μ, length(inds))

log_lm(lm::TemplateModel) = false
log_lm(lm::LinearModel) = lm.log
Base.getindex(lm::LinearModel, s::Symbol) = getfield(lm, s)
Base.eachindex(lm::TemplateModel) = fieldnames(typeof(lm))
Base.eachindex(lm::LinearModel) = fieldnames(typeof(lm))[1:end-1]  # dealing with log fieldname
Base.setindex!(lm::LinearModel, a::AbstractVecOrMat, s::Symbol) = (lm[s] .= a)
vec(lm::LinearModel) = [lm[i] for i in eachindex(lm)]
vec(lm::TemplateModel) = [lm.μ]
vec(lms::Vector{<:LinearModel}) = [vec(lm) for lm in lms]

LinearModel(M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector; log_lm::Bool=_log_lm_default) = FullLinearModel(M, s, μ, log)
LinearModel(M::AbstractMatrix, s::AbstractMatrix; log_lm::Bool=_log_lm_default) = BaseLinearModel(M, s, log)
LinearModel(μ::AbstractVector, n::Int) = TemplateModel(μ, n)

LinearModel(lm::FullLinearModel, s::AbstractMatrix) = FullLinearModel(lm.M, s, lm.μ, log)
LinearModel(lm::BaseLinearModel, s::AbstractMatrix) = BaseLinearModel(lm.M, s, lm.log)
LinearModel(lm::TemplateModel, s::AbstractMatrix) = lm

# Ref(lm::FullLinearModel) = [Ref(lm.M), Ref(lm.s), Ref(lm.μ)]
# Ref(lm::BaseLinearModel) = [Ref(lm.M), Ref(lm.s)]
# Ref(lm::TemplateModel) = [Ref(lm.μ)]

# _eval_lm(μ, n::Int) = repeat(μ, 1, n)
_eval_lm(μ, n::Int) = μ * ones(n)'  # this is faster I dont know why
_eval_lm(M, s; log_lm::Bool=false) = log_lm ? (return exp.(M * s)) : (return (M * s))
_eval_lm(M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector) = muladd(M, s, μ)  # faster, but Nabla doesn't handle it
_eval_lm(M, s, μ; log_lm::Bool=false) = log_lm ? (return exp.(M * s) .* μ) : (return (M * s) .+ μ)

_eval_lm(flm::FullLinearModel) = _eval_lm(flm.M, flm.s, flm.μ; log_lm=flm.log)
_eval_lm(blm::BaseLinearModel) = _eval_lm(blm.M, blm.s; log_lm=blm.log)
_eval_lm(tlm::TemplateModel) = _eval_lm(tlm.μ, tlm.n)
(lm::LinearModel)() = _eval_lm(lm)

(flm::FullLinearModel)(inds::AbstractVecOrMat) = _eval_lm(view(flm.M, inds, :), flm.s, flm.μ; log_lm=flm.log)
(blm::BaseLinearModel)(inds::AbstractVecOrMat) = _eval_lm(view(blm.M, inds, :), blm.s; log_lm=blm.log)
(tlm::TemplateModel)(inds::AbstractVecOrMat) = repeat(view(tlm.μ, inds), 1, tlm.n)

function copy_to_LinearModel!(to::TemplateModel, from::LinearModel)
	to.μ .= from.μ
end
function copy_to_LinearModel!(to::LinearModel, from::LinearModel)
	@assert typeof(to) == typeof(from)
	@assert to.log == from.log
	for i in eachindex(from)
		getfield(to, i) .= getfield(from, i)
	end
end
copy_to_LinearModel!(to::TemplateModel, from::LinearModel, inds) =
	copy_to_LinearModel!(to, from)
function copy_to_LinearModel!(to::FullLinearModel, from::FullLinearModel, inds)
	@assert to.log == from.log
	to.μ .= from.μ
	to.M .= view(from.M, :, inds)
	to.s .= view(from.s, inds, :)
end
function copy_to_LinearModel!(to::LinearModel, from::Vector)
	if typeof(to) <: TemplateModel
		if typeof(from) <: Vector{<:Real}
			to.μ .= from
		else
			@assert length(from) == 1
			to.μ .= from[1]
		end
	else
		fns = eachindex(to)
		@assert length(from) == length(fns)
		for i in eachindex(fns)
			getfield(to, fns[i]) .= from[i]
		end
	end
end

struct Submodel{T<:Number, AV1<:AbstractVector{T}, AV2<:AbstractVector{T}, AA<:AbstractArray{T}}
    log_λ::AV1
    λ::AV2
	lm::LinearModel
	A_sde::StaticMatrix
	Σ_sde::StaticMatrix
	Δℓ_coeff::AA
end
function Submodel(log_λ_obs::AbstractVecOrMat, n_comp::Int, log_λ_gp::Real; include_mean::Bool=true, log_lm::Bool=_log_lm_default, kwargs...)
	@assert type in _acceptable_types
	n_obs = size(log_λ_obs, 2)
	log_λ, λ = create_λ_template(log_λ_obs; kwargs...)
	len = length(log_λ)
	if include_mean
		if n_comp > 0
			lm = FullLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs), ones(len), log_lm)
		else
			lm = TemplateModel(ones(len), n_obs)
		end
	else
		if n_comp > 0
			lm = BaseLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs), log_lm)
		end
		@error "you need a mean if you don't want any components"
	end
	temporal_gps_λ = 1 / log_λ_gp
	A_sde, Σ_sde = gp_sde_prediction_matrices(step(log_λ), temporal_gps_λ)
	sparsity = Int(round(0.5 / (step(log_λ) * temporal_gps_λ)))
	Δℓ_coeff = gp_Δℓ_coefficients(length(log_λ), A_sde, Σ_sde; sparsity=sparsity)
	return Submodel(log_λ, λ, lm, A_sde, Σ_sde, Δℓ_coeff)
end
function Submodel(log_λ::AV1, λ::AV2, lm, A_sde::StaticMatrix, Σ_sde::StaticMatrix, Δℓ_coeff::AA) where {T<:Number, AV1<:AbstractVector{T}, AV2<:AbstractVector{T}, AA<:AbstractArray{T}}
	if typeof(lm) <: TemplateModel
		@assert length(log_λ) == length(λ) == length(lm.μ) == size(Δℓ_coeff, 1) == size(Δℓ_coeff, 2)
	else
		@assert length(log_λ) == length(λ) == size(lm.M, 1) == size(Δℓ_coeff, 1) == size(Δℓ_coeff, 2)
	end
	@assert size(A_sde) == size(Σ_sde)
	return Submodel{T, AV1, AV2}(log_λ, λ, lm, A_sde, Σ_sde, Δℓ_coeff)
end
(sm::Submodel)(inds::AbstractVecOrMat) =
	Submodel(sm.log_λ, sm.λ, LinearModel(sm.lm, inds), sm.A_sde, sm.Σ_sde, sm.Δℓ_coeff)
Base.copy(sm::Submodel) = Submodel(sm.log_λ, sm.λ, copy(sm.lm), sm.A_sde, sm.Σ_sde, sm.Δℓ_coeff)

function _shift_log_λ_model(log_λ_obs_from, log_λ_obs_to, log_λ_model_from)
	n_obs = size(log_λ_obs_from, 2)
	dop = [log_λ_obs_from[1, i] - log_λ_obs_to[1, i] for i in 1:n_obs]
	log_λ_model_to = ones(length(log_λ_model_from), n_obs)
	for i in 1:n_obs
		log_λ_model_to[:, i] .= log_λ_model_from .+ dop[i]
	end
	return log_λ_model_to
end

# They need to be different or else the stellar μ will be surpressed
# We only use the ones that have seemed useful
default_reg_tel = Dict([(:GP_μ, 1e6), (:L2_μ, 1e6), (:L1_μ, 1e5), (:L1_μ₊_factor, 6.),
	(:GP_M, 1e7), (:L1_M, 1e7)])
default_reg_star = Dict([(:GP_μ, 1e2), (:L2_μ, 1e-2), (:L1_μ, 1e2), (:L1_μ₊_factor, 6.),
	(:GP_M, 1e4), (:L1_M, 1e7)])
default_reg_tel_full = Dict([(:GP_μ, 1e6), (:L2_μ, 1e6), (:L1_μ, 1e5),
	(:L1_μ₊_factor, 6.), (:GP_M, 1e7), (:L2_M, 1e4), (:L1_M, 1e7)])
default_reg_star_full = Dict([(:GP_μ, 1e2), (:L2_μ, 1e-2), (:L1_μ, 1e1),
	(:L1_μ₊_factor, 6.), (:GP_M, 1e4), (:L2_M, 1e4), (:L1_M, 1e7)])

function oversamp_interp_helper(to_bounds::AbstractVector, from_x::AbstractVector)
	ans = spzeros(length(to_bounds)-1, length(from_x))
	bounds_inds = searchsortednearest(from_x, to_bounds)
	for i in 1:size(ans, 1)
		x_lo, x_hi = to_bounds[i], to_bounds[i+1]  # values of bounds
		lo_ind, hi_ind = bounds_inds[i], bounds_inds[i+1]  # indices of points in model closest to the bounds

		# if necessary, shrink so that so from_x[lo_ind] and from_x[hi_ind] are in the bounds
		if from_x[lo_ind] < x_lo; lo_ind += 1 end
		if from_x[hi_ind] > x_hi; hi_ind -= 1 end

		edge_term_lo = (from_x[lo_ind] - x_lo) ^ 2 / (from_x[lo_ind] - from_x[lo_ind-1])
		edge_term_hi = (x_hi - from_x[hi_ind]) ^ 2 / (from_x[hi_ind+1] - from_x[hi_ind])

		ans[i, lo_ind-1] = edge_term_lo
		ans[i, lo_ind] = from_x[lo_ind+1] + from_x[lo_ind] - 2 * x_lo - edge_term_lo

		ans[i, lo_ind+1:hi_ind-1] .= view(from_x, lo_ind+2:hi_ind) .- view(from_x, lo_ind:hi_ind-2)

		ans[i, hi_ind] = 2 * x_hi - from_x[hi_ind] - from_x[hi_ind-1] - edge_term_hi
		ans[i, hi_ind+1] = edge_term_hi
		# println(sum(view(ans, i, lo_ind-1:hi_ind+1))," vs ", 2 * (x_hi - x_lo))
		# @assert isapprox(sum(view(ans, i, lo_ind-1:hi_ind+1)), 2 * (x_hi - x_lo); rtol=1e-3)
		ans[i, lo_ind-1:hi_ind+1] ./= sum(view(ans, i, lo_ind-1:hi_ind+1))
		# ans[i, lo_ind-1:hi_ind+1] ./= 2 * (x_hi - x_lo)
	end
	dropzeros!(ans)
	return ans
end
oversamp_interp_helper(to_bounds::AbstractMatrix, from_x::AbstractVector) =
	[oversamp_interp_helper(view(to_bounds, :, i), from_x) for i in 1:size(to_bounds, 2)]

function undersamp_interp_helper(to_x::AbstractVector, from_x::AbstractVector)
	ans = spzeros(length(to_x), length(from_x))
	# ans = sparse(Float64[],Float64[],Float64[],length(to_x),length(from_x))
	to_inds = searchsortednearest(from_x, to_x; lower=true)
	for i in 1:size(ans, 1)
		x_new = to_x[i]
		ind = to_inds[i]  # index of point in model below to_x[i]
		dif = (x_new-from_x[ind]) / (from_x[ind+1] - from_x[ind])
		ans[i, ind] = 1 - dif
		ans[i, ind + 1] = dif
	end
	dropzeros!(ans)
	return ans
end
undersamp_interp_helper(to_x::AbstractMatrix, from_x::AbstractVector) =
	[undersamp_interp_helper(view(to_x, :, i), from_x) for i in 1:size(to_x, 2)]

struct OrderModelDPCA{T<:Number} <: OrderModel
    tel::Submodel
    star::Submodel
	rv::Submodel
	reg_tel::Dict{Symbol, T}
	reg_star::Dict{Symbol, T}
	b2o::AbstractVector{<:_current_matrix_modifier}
	t2o::AbstractVector{<:_current_matrix_modifier}
	metadata::Dict{Symbol, Any}
	n::Int
end
struct OrderModelWobble{T<:Number} <: OrderModel
    tel::Submodel
    star::Submodel
	rv::AbstractVector
	reg_tel::Dict{Symbol, T}
	reg_star::Dict{Symbol, T}
	b2o::StellarInterpolationHelper
	bary_rvs::AbstractVector{<:Real}
	t2o::AbstractVector{<:_current_matrix_modifier}
	metadata::Dict{Symbol, Any}
	n::Int
end
function OrderModel(
	d::Data,
	instrument::String,
	order::Int,
	star::String;
	n_comp_tel::Int=2,
	n_comp_star::Int=2,
	oversamp::Bool=true,
	dpca::Bool=true,
	log_λ_gp_star::Real=1/SOAP_gp_params.λ,
	log_λ_gp_tel::Real=1/LSF_gp_params.λ,
	kwargs...)

	tel = Submodel(d.log_λ_obs, n_comp_tel, log_λ_gp_tel; kwargs...)
	star = Submodel(d.log_λ_star, n_comp_star, log_λ_gp_star; kwargs...)
	n_obs = size(d.log_λ_obs, 2)
	dpca ?
		rv = Submodel(d.log_λ_star, 1, log_λ_gp_star; include_mean=false, kwargs...) :
		rv = zeros(n_obs)

	bary_rvs = D_to_rv.([median(d.log_λ_star[:, i] - d.log_λ_obs[:, i]) for i in 1:n_obs])
	todo = Dict([(:initialized, false), (:reg_improved, false), (:downsized, false), (:err_estimated, false)])
	metadata = Dict([(:todo, todo), (:instrument, instrument), (:order, order), (:star, star)])
	if dpca
		if oversamp
			b2o = oversamp_interp_helper(d.log_λ_star_bounds, star.log_λ)
			t2o = oversamp_interp_helper(d.log_λ_obs_bounds, tel.log_λ)
		else
			b2o = undersamp_interp_helper(d.log_λ_star, star.log_λ)
			t2o = undersamp_interp_helper(d.log_λ_obs, tel.log_λ)
		end
		return OrderModelDPCA(tel, star, rv, copy(default_reg_tel), copy(default_reg_star), b2o, t2o, metadata, n_obs)
	else
		b2o = StellarInterpolationHelper(star.log_λ, bary_rvs, d.log_λ_obs)
		if oversamp
			t2o = oversamp_interp_helper(d.log_λ_obs_bounds, tel.log_λ)
		else
			t2o = undersamp_interp_helper(d.log_λ_obs, tel.log_λ)
		end
		return OrderModelWobble(tel, star, rv, copy(default_reg_tel), copy(default_reg_star), b2o, bary_rvs, t2o, metadata, n_obs)
	end
end
Base.copy(om::OrderModelDPCA) = OrderModelDPCA(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), om.b2o, om.t2o, copy(om.metadata), om.n)
(om::OrderModelDPCA)(inds::AbstractVecOrMat) =
	OrderModelDPCA(om.tel(inds), om.star(inds), om.rv(inds), copy(om.reg_tel),
		copy(om.reg_star), view(om.b2o, inds), view(om.t2o, inds), copy(om.metadata), length(inds))
Base.copy(om::OrderModelWobble) = OrderModelWobble(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), om.b2o, om.bary_rvs, om.t2o, copy(om.metadata), om.n)
(om::OrderModelWobble)(inds::AbstractVecOrMat) =
	OrderModelWobble(om.tel(inds), om.star(inds), view(om.rv, inds), copy(om.reg_tel),
		copy(om.reg_star), om.b2o(inds, length(om.star.lm.μ)), view(om.bary_rvs, inds), view(om.t2o, inds), copy(om.metadata), length(inds))

function rm_dict!(d::Dict)
	for (key, value) in d
		delete!(d, key)
	end
end
function rm_regularization!(om::OrderModel)
	rm_dict!(om.reg_tel)
	rm_dict!(om.reg_star)
end
function zero_regularization(om::OrderModel; include_L1_factor::Bool=false)
	for (key, value) in om.reg_tel
		if include_L1_factor || (key != :L1_μ₊_factor)
			om.reg_tel[key] = 0
		end
	end
	for (key, value) in om.reg_star
		if include_L1_factor || (key != :L1_μ₊_factor)
			om.reg_star[key] = 0
		end
	end
end
function reset_regularization!(om::OrderModel)
	for (key, value) in om.reg_tel
		om.reg_tel[key] = default_reg_tel_full[key]
	end
	for (key, value) in om.reg_star
		om.reg_star[key] = default_reg_star_full[key]
	end
end

function _eval_lm_vec(om::OrderModel, v; log_lm::Bool=_log_lm_default)
	@assert 0 < length(v) < 4
	if length(v)==1
		return _eval_lm(v[1], om.n)
	elseif length(v)==2
		return _eval_lm(v[1], v[2]; log_lm=log_lm)
	elseif length(v)==3
		return _eval_lm(v[1], v[2], v[3]; log_lm=log_lm)
	end
end

# I have no idea why the negative sign needs to be here
rvs(model::OrderModelDPCA) = vec(model.rv.lm.s .* -light_speed_nu)
rvs(model::OrderModelWobble) = model.rv

function downsize(lm::FullLinearModel, n_comp::Int)
	if n_comp > 0
		return FullLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :], copy(lm.μ), lm.log)
	else
		return TemplateModel(copy(lm.μ), size(lm.s, 2))
	end
end
downsize(lm::BaseLinearModel, n_comp::Int) =
	BaseLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :], lm.log)
function downsize(lm::TemplateModel, n_comp::Int)
	@assert n_comp==0
	return TemplateModel(copy(lm.μ), lm.n)
end
downsize(sm::Submodel, n_comp::Int) =
	Submodel(copy(sm.log_λ), copy(sm.λ), downsize(sm.lm, n_comp), copy(sm.A_sde), copy(sm.Σ_sde), copy(sm.Δℓ_coeff))
downsize(m::OrderModelDPCA, n_comp_tel::Int, n_comp_star::Int) =
	OrderModelDPCA(
		downsize(m.tel, n_comp_tel),
		downsize(m.star, n_comp_star),
		copy(m.rv), copy(m.reg_tel), copy(m.reg_star), m.b2o, m.t2o, copy(m.metadata), m.n)
downsize(m::OrderModelWobble, n_comp_tel::Int, n_comp_star::Int) =
	OrderModelWobble(
		downsize(m.tel, n_comp_tel),
		downsize(m.star, n_comp_star),
		copy(m.rv), copy(m.reg_tel), copy(m.reg_star), m.b2o, m.bary_rvs, m.t2o, copy(m.metadata), m.n)

spectra_interp(model::AbstractVector, interp_helper::_current_matrix_modifier) =
	interp_helper * model
spectra_interp(model::AbstractMatrix, interp_helper::AbstractVector{<:_current_matrix_modifier}) =
	hcat([spectra_interp(view(model, :, i), interp_helper[i]) for i in 1:size(model, 2)]...)
# spectra_interp_nabla(model, interp_helper::AbstractVector{<:_current_matrix_modifier}) =
# 	hcat([interp_helper[i] * model[:, i] for i in 1:size(model, 2)]...)
# spectra_interp(model, interp_helper::AbstractVector{<:_current_matrix_modifier}) =
# 	spectra_interp_nabla(model, interp_helper)
@explicit_intercepts spectra_interp Tuple{AbstractMatrix, AbstractVector{<:_current_matrix_modifier}} [true, false]
Nabla.∇(::typeof(spectra_interp), ::Type{Arg{1}}, _, y, ȳ, model, interp_helper) =
	hcat([interp_helper[i]' * view(ȳ, :, i) for i in 1:size(model, 2)]...)

tel_model(om::OrderModel; lm=om.tel.lm::LinearModel) = spectra_interp(lm(), om.t2o)
star_model(om::OrderModelDPCA; lm=om.star.lm::LinearModel) = spectra_interp(lm(), om.b2o)
rv_model(om::OrderModelDPCA; lm=om.rv.lm::LinearModel) = spectra_interp(lm(), om.b2o)
star_model(om::OrderModelWobble; lm=om.star.lm::LinearModel) = spectra_interp(lm(), om.rv .+ om.bary_rvs, om.b2o)

# function fix_FullLinearModel_s!(flm, min::Number, max::Number)
# 	@assert all(min .< flm.μ .< max)
# 	result = ones(typeof(flm.μ[1]), length(flm.μ))
# 	for i in 1:size(flm.s, 2)
# 		result[:] = _eval_lm(flm.M, flm.s[:, i], flm.μ)
# 		while any(result .> max) || any(result .< min)
# 			# println("$i, old s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
# 			flm.s[:, i] ./= 2
# 			result[:] = _eval_lm(flm.M, view(flm.s, :, i), flm.μ)
# 			# println("$i, new s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
# 		end
# 	end
# end

function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    gp_post = posterior(finite_GP, ys)
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end

function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    return mean.(get_marginal_GP(finite_GP, ys, xs))
end

function build_gp(params::NamedTuple)
	f_naive = AbstractGPs.GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
	return to_sde(f_naive, SArrayStorage(Float64))
end
SOAP_gp_params = (var_kernel = 0.19222435463373258, λ = 26801.464367577082)
SOAP_gp = build_gp(SOAP_gp_params)
SOAP_gp_var = 1e-6
LSF_gp_params = (var_kernel = 0.2, λ = 185325.)
LSF_gp = build_gp(LSF_gp_params)
LSF_gp_var = 1e-6

# ParameterHandling version
# SOAP_gp_params = (var_kernel = positive(3.3270754364467443), λ = positive(1 / 9.021560480866474e-5))
# flat_SOAP_gp_params, unflatten = value_flatten(SOAP_gp_params)
# # unflatten(flat_SOAP_gp_params) == ParameterHandling.value(SOAP_gp_params)  # true
# SOAP_gp = build_gp(ParameterHandling.value(SOAP_gp_params))
function _spectra_interp_gp!(fluxes::AbstractVector, log_λ, flux_obs::AbstractVector, var_obs, log_λ_obs; gp_mean::Number=0., gp_base=SOAP_gp)
	gp = get_marginal_GP(gp_base(log_λ_obs, var_obs), flux_obs .- gp_mean, log_λ)
	fluxes[:] = mean.(gp) .+ gp_mean
	return gp
end
function _spectra_interp_gp!(fluxes::AbstractVector, vars, log_λ, flux_obs::AbstractVector, var_obs, log_λ_obs; keep_mask::Bool=true, kwargs...)
	gp = _spectra_interp_gp!(fluxes, log_λ, flux_obs, var_obs, log_λ_obs; kwargs...)
	vars[:] = var.(gp)
	if keep_mask
		inds = searchsortednearest(log_λ_obs, log_λ; lower=true)
		# for i in 1:length(inds)
		for i in 1:length(log_λ)
			if log_λ[i] <= log_λ_obs[1]
				if isinf(var_obs[1]); vars[i] = Inf end
			elseif log_λ[i] >= log_λ_obs[end]
				if isinf(var_obs[end]); vars[i] = Inf end
			elseif isinf(var_obs[inds[i]]) && isinf(var_obs[inds[i]+1])
				vars[i] = Inf
			end
		end
	end
	return gp
end
function _spectra_interp_gp!(fluxes::AbstractMatrix, log_λ, flux_obs, var_obs, log_λ_obs; kwargs...)
	for i in 1:size(flux_obs, 2)
		_spectra_interp_gp!(view(fluxes, :, i), log_λ, view(flux_obs, :, i), view(var_obs, :, i), view(log_λ_obs, :, i); kwargs...)
	end
end
function _spectra_interp_gp!(fluxes::AbstractMatrix, vars, log_λ, flux_obs, var_obs, log_λ_obs; kwargs...)
	for i in 1:size(flux_obs, 2)
		_spectra_interp_gp!(view(fluxes, :, i), view(vars, :, i), log_λ, view(flux_obs, :, i), view(var_obs, :, i), view(log_λ_obs, :, i); kwargs...)
	end
end

function _spectra_interp_gp_div_gp!(fluxes::AbstractMatrix, vars::AbstractMatrix, log_λ::AbstractVector, flux_obs::AbstractMatrix, var_obs::AbstractMatrix, log_λ_obs::AbstractMatrix, flux_other::AbstractMatrix, var_other::AbstractMatrix, log_λ_other::AbstractMatrix; gp_mean::Number=1., gp_base=SOAP_gp, gp_var=SOAP_gp_var, keep_mask::Bool=true, return_weights::Bool=false)
	return_weights ? _inf = 0 : _inf = Inf
	for i in 1:size(flux_obs, 2)
		gpn = get_marginal_GP(gp_base(view(log_λ_obs, :, i), view(var_obs, :, i) .+ gp_var), view(flux_obs, :, i) .- gp_mean, log_λ)
		gpd = get_marginal_GP(gp_base(view(log_λ_other, :, i), view(var_other, :, i) .+ gp_var), view(flux_other, :, i) .- gp_mean, log_λ)
		gpn_μ = mean.(gpn) .+ gp_mean
		gpd_μ = mean.(gpd) .+ gp_mean
		fluxes[:, i] .= gpn_μ ./ gpd_μ
		return_weights ?
			vars[:, i] .= 1 ./ (var.(gpn) .+ ((gpn_μ .^ 2 .* var.(gpd)) ./ (gpd_μ .^2))) ./ (gpd_μ .^2) :
			vars[:, i] .= (var.(gpn) .+ ((gpn_μ .^ 2 .* var.(gpd)) ./ (gpd_μ .^2))) ./ (gpd_μ .^2)
		if keep_mask
			inds = searchsortednearest(view(log_λ_obs, :, i), log_λ; lower=true)
			for j in 1:length(inds)
				if log_λ[j] <= log_λ_obs[1, i]
					if isinf(var_obs[1, i]); vars[j, i] = _inf end
				elseif log_λ[j] >= log_λ_obs[end, i]
					if isinf(var_obs[end, i]); vars[j, i] = _inf end
				elseif isinf(var_obs[inds[j], i]) && isinf(var_obs[inds[j]+1, i])
					vars[j, i] = _inf
				end
			end
		end
	end
end
function _spectra_interp_gp_div_gp(log_λ::AbstractVector, flux_obs::AbstractMatrix, var_obs::AbstractMatrix, log_λ_obs::AbstractMatrix, flux_other::AbstractMatrix, var_other::AbstractMatrix, log_λ_other::AbstractMatrix; kwargs...)
	fluxes = Array{Float64}(undef, length(log_λ), size(flux_obs, 2))
	vars = Array{Float64}(undef, length(log_λ), size(flux_obs, 2))
	 _spectra_interp_gp_div_gp!(fluxes, vars, log_λ, flux_obs, var_obs, log_λ_obs, flux_other, var_other, log_λ_other; kwargs...)
	return fluxes, vars
end

# function n_comps_needed(sm::Submodel; threshold::Real=0.05)
#     @assert 0 < threshold < 1
#     s_var = sum(abs2, sm.lm.s; dims=2)
#     return findfirst(s_var ./ sum(s_var) .< threshold)[1] - 1
# end

function initializations!(om::OrderModel, d::Data; μ_min::Number=0, μ_max::Number=1.25,
	seed::Union{OrderModel, Nothing}=nothing, use_mean::Bool=true, multithread::Bool=nthreads() > 3, remove_reciprocal_continuum::Bool=false, pairwise::Bool=true)

	seeded = !isnothing(seed)

	n_obs = size(d.flux, 2)
	is_time_variable(om.star.lm) ? n_comp_star = size(om.star.lm.M, 2) : n_comp_star = 0
	is_time_variable(om.tel.lm) ? n_comp_tel = size(om.tel.lm.M, 2) : n_comp_tel = 0

	star_log_λ_tel = _shift_log_λ_model(d.log_λ_obs, d.log_λ_star, om.star.log_λ)
	tel_log_λ_star = _shift_log_λ_model(d.log_λ_star, d.log_λ_obs, om.tel.log_λ)
	flux_star = ones(length(om.star.log_λ), n_obs)
	vars_star = SOAP_gp_var .* ones(length(om.star.log_λ), n_obs)
	flux_tel = ones(length(om.tel.log_λ), n_obs)
	vars_tel = SOAP_gp_var .* ones(length(om.tel.log_λ), n_obs)
	lm_tel = copy(om.tel.lm)
	lm_star = FullLinearModel(zeros(length(om.star.lm.μ), n_comp_star + 1), zeros(n_comp_star + 1, n_obs), zeros(length(om.star.lm.μ)), log_lm(om.star.lm))


	function reciprocal_continuum_mask(continuum::AbstractVector, other_interpolated_continuum::AbstractVector; probe_depth::Real=0.02, return_cc::Bool=false)
		# probe_depth=0.0
		cc = (1 .- continuum) .* (1 .- other_interpolated_continuum)
		ccm = find_modes(-cc)

		ccm = [i for i in ccm if ((cc[i] < -(probe_depth^2)) && (0.5 < abs(continuum[i] / other_interpolated_continuum[i]) < 2))]
		mask = zeros(Bool, length(cc))
		l = length(cc)
		for m in ccm
			i = m
			while i <= l && cc[i] < 0
				mask[i] = true
				i += 1
			end
			i = m-1
			while i >= 1 && cc[i] < 0
				mask[i] = true
				i -= 1
			end
		end
		if return_cc
			return mask, cc
		end
		return mask
	end
	reciprocal_continuum_mask(continuum::AbstractVector, other_interpolated_continuum::AbstractMatrix; kwargs...) =
		reciprocal_continuum_mask(continuum, vec(mean(other_interpolated_continuum; dims=2)); kwargs...)

	function remove_reciprocal_continuum!(lm_star::LinearModel, lm_tel::LinearModel, flux_star_holder::AbstractMatrix, vars_star_holder::AbstractMatrix, flux_tel_holder::AbstractMatrix, vars_tel_holder::AbstractMatrix; use_stellar_continuum::Bool=true, kwargs...)

		_, c_t, _ = calc_continuum(om.tel.λ, lm_tel.μ, ones(length(lm_tel.μ)) ./ 1000;
			min_R_factor=1, smoothing_half_width=0,
			stretch_factor=10., merging_threshold = 0.)

		_, c_s, _ = calc_continuum(om.star.λ, lm_star.μ, ones(length(lm_star.μ)) ./ 1000;
			min_R_factor=1,
			stretch_factor=10., merging_threshold = 0.)

		flux_star_holder .= c_s
		vars_star_holder .= SOAP_gp_var
		_spectra_interp_gp!(flux_tel_holder, vars_tel_holder, om.tel.log_λ, flux_star_holder, vars_star_holder, star_log_λ_tel; gp_mean=1.)
		m, cc = reciprocal_continuum_mask(c_t, flux_tel_holder; return_cc=true, kwargs...)
		use_stellar_continuum ?
			lm_tel.μ[m] .*= vec(mean(flux_tel_holder[m, :]; dims=2)) :
			lm_tel.μ[m] ./= c_t[m]
		did_anything = any(m)

		flux_tel .= c_t
		vars_tel .= SOAP_gp_var
		_spectra_interp_gp!(flux_star_holder, vars_star_holder, om.star.log_λ, flux_tel_holder, vars_tel_holder, tel_log_λ_star; gp_mean=1.)
		m, cc = reciprocal_continuum_mask(c_s, flux_star_holder; return_cc=true, kwargs...)
		use_stellar_continuum ?
			lm_star.μ[m] ./= c_s[m] :
			lm_star.μ[m] .*= vec(mean(flux_star_holder[m, :]; dims=2))
		return did_anything || any(m)

	end

	function update_tel!(lm_star::LinearModel, lm_tel::LinearModel, use_mean::Bool; remove_reciprocal_continuum::Bool=false)
		_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
		lm_tel.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)
		if remove_reciprocal_continuum
			did_anything = remove_reciprocal_continuum!(lm_star, lm_tel, flux_star, vars_star_copy, flux_tel, vars_tel_copy)
			vars_star_copy .= vars_star
			vars_tel_copy .= vars_tel
			flux_star .= lm_star.μ
			if did_anything
				_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
				lm_tel.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)
			end
		end
		mask_low_pixels!(flux_tel, vars_tel)
		mask_high_pixels!(flux_tel, vars_tel)
		if is_time_variable(om.tel)
			EMPCA!(lm_tel.M, lm_tel.s, lm_tel.μ, flux_tel, 1 ./ vars_tel; log_lm=log_lm(lm_tel))
		end
	end

	function update_star!(lm_star::LinearModel, lm_tel::LinearModel, use_mean::Bool; remove_reciprocal_continuum::Bool=false)
		_spectra_interp_gp_div_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)
		lm_star.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=use_mean)
		if remove_reciprocal_continuum
			did_anything = remove_reciprocal_continuum!(lm_star, lm_tel, flux_star, vars_star_copy, flux_tel, vars_tel_copy)
			vars_star_copy .= vars_star
			vars_tel_copy .= vars_tel
			if did_anything
				flux_tel .= lm_tel.μ
				_spectra_interp_gp_div_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)
				lm_star.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=use_mean)
			end
		end
		mask_low_pixels!(flux_star, vars_star)
		mask_high_pixels!(flux_star, vars_star)
		DEMPCA!(lm_star.M, lm_star.s, lm_star.μ, flux_star, 1 ./ vars_star, calc_doppler_component_RVSKL(om.star.λ, lm_star.μ); log_lm=log_lm(lm_star))
	end

	# if we have a seed model to get tellurics from
	# TODO should we also transfer basis vectors?
	if seeded

		# get initial interpolated guess for telluric template
		_spectra_interp_gp!(lm_tel.μ, om.tel.log_λ, seed.tel.lm.μ, LSF_gp_var, seed.tel.log_λ; gp_base=LSF_gp, gp_mean=1.)

		# finding where the telluric lines are in the seeded template
		tel_μ_interp = 1 .- vec(spectra_interp(lm_tel.μ, om.t2o))
		mask = tel_μ_interp .> 0.05
		step = (d.log_λ_obs[end, 1] - d.log_λ_obs[1, 1]) / (size(d.log_λ_obs, 1) - 1)
		width = Int(floor(2.5e-5 / step))
		lo = 0
		inds = UnitRange[]
		for i in 1:length(mask)
			if mask[i] == 1
				if lo == 0
					lo = i
				end
				if i==length(mask) || mask[i+1] == 0
					append!(inds, [maximum([1, lo-width]):minimum([i+width, length(mask)])])
					lo = 0
				end
			end
		end

		# finding the right template amplitude
		χ²_reduce(i) = (sum(d.flux[i, :] ./ d.var[i, :]; dims=1) ./ sum(1 ./ d.var[i, :]; dims=1))[1]
		scaling(j, y1, y2, x_width) = y1 + (y2 - y1) / (x_width - 1) * (j - 1)
		proposed_amps = zeros(length(inds))
		proposed_χ²_per_pixel = zeros(length(inds))
		for i in 1:length(inds)
			ind = inds[i]
			tel_μ_test = repeat(tel_μ_interp[ind], n_obs)
			d_test_flux = collect(Iterators.flatten(d.flux[ind, :]))
			d_test_var = collect(Iterators.flatten(d.var[ind, :]))
			y1, y2 = χ²_reduce(ind[1]), χ²_reduce(ind[end])

			scales = [scaling(j, y1, y2, length(ind)) for j in 1:length(ind)]
			for i in 1:n_obs
				d_test_flux[(1+(i-1)*length(ind)):(i*length(ind))] ./= scales
				d_test_var[(1+(i-1)*length(ind)):(i*length(ind))] ./= scales .* scales
			end
			d_test_flux[:] = 1 .- d_test_flux
			proposed_amps[i] = general_lst_sq(reshape(tel_μ_test,(length(tel_μ_test), 1)), d_test_flux, Diagonal(d_test_var))[1]
			proposed_χ²_per_pixel[i] = sum(_χ²_loss(proposed_amps[i] * tel_μ_test, d_test_flux, d_test_var)) / length(d_test_flux)
		end
		good_amps = proposed_amps[proposed_χ²_per_pixel .< 4e4]
		if length(good_amps) < 1
			@warn "insufficient good telluric fits to estimate an amplitude. Defaulting to no scaling"
		else
			lm_tel.μ[:] = 1 .+ (median(good_amps) .* (lm_tel.μ .- 1))
		end

		# stellar flux dividing out template tellurics
		flux_tel .= lm_tel.μ
		_spectra_interp_gp_div_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)

		om = downsize(om, size(seed.tel.lm.M, 2), n_comp_star)
		lm_tel = downsize(lm_tel, size(seed.tel.lm.M, 2))

	else
		# stellar flux assuming no tellurics
		_spectra_interp_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var .+ SOAP_gp_var, d.log_λ_star; gp_mean=1.)
	end
	vars_star_copy = copy(vars_star)
	# stellar template assuming no tellurics
	lm_star.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=seeded)

	if !seeded
		star_template_χ² = sum(_χ²_loss(star_model(om; lm=lm_star), d))

		# telluric template assuming no stellar
		_spectra_interp_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var .+ SOAP_gp_var, d.log_λ_obs; gp_mean=1.)
		lm_tel.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=seeded)
		tel_template_χ² = sum(_χ²_loss(tel_model(om; lm=lm_tel), d))
	end
	vars_tel_copy = copy(vars_tel)

	stellar_dominated = seeded || tel_template_χ² > star_template_χ²
	# use the better model as the initial basis
	if stellar_dominated
		println("using stellar model as initial template")
		# telluric model with stellar template
		flux_star .= lm_star.μ
		update_tel!(lm_star, lm_tel, seeded||use_mean; remove_reciprocal_continuum=remove_reciprocal_continuum)
		# updating stellar model with modified telluric template
		flux_tel .= lm_tel.μ
		update_star!(lm_star, lm_tel, seeded||use_mean)
	else
		println("using telluric model as initial template")
		# stellar model with telluric template
		flux_tel .= lm_tel.μ
		update_star!(lm_star, lm_tel, seeded||use_mean)
		# updating telluric model with modified stellar template
		flux_star .= lm_star.μ
		update_tel!(lm_star, lm_tel, seeded||use_mean)
	end

	# stellar models with n basis telluric model
	if is_time_variable(om.tel)
		if pairwise
			lm_star = [copy(lm_star) for i in 1:(n_comp_tel+1)]
			if multithread
				# @threads for i in 1:n_comp_tel
				ThreadsX.foreach(1:n_comp_tel) do i
					_flux_star, _vars_star = _spectra_interp_gp_div_gp(om.star.log_λ, d.flux, d.var, d.log_λ_star, _eval_lm(view(lm_tel.M, :, 1:i), view(lm_tel.s, 1:i, :), lm_tel.μ; log_lm=log_lm(lm_tel)), vars_tel, tel_log_λ_star)
					lm_star[i+1].μ[:] = make_template(_flux_star, _vars_star; min=μ_min, max=μ_max, use_mean=true)
					mask_low_pixels!(_flux_star, _vars_star)
					mask_high_pixels!(_flux_star, _vars_star)
					DEMPCA!(lm_star[i+1].M, lm_star[i+1].s, lm_star[i+1].μ, _flux_star, 1 ./ _vars_star, calc_doppler_component_RVSKL(om.star.λ, lm_star[i+1].μ); log_lm=log_lm(lm_star[i+1]))
				end
			else
				for i in 1:n_comp_tel
					flux_tel .= _eval_lm(view(lm_tel.M, :, 1:i), view(lm_tel.s, 1:i, :), lm_tel.μ; log_lm=log_lm(lm_tel))
					update_star!(lm_star[i+1], lm_tel, true)
				end
			end
		else
			lm_star = [copy(lm_star) for i in 1:2]
			flux_tel .= _eval_lm(view(lm_tel.M, :, :), view(lm_tel.s, :, :), lm_tel.μ; log_lm=log_lm(lm_tel))
			update_star!(lm_star[2], lm_tel, true)
		end
	else
		lm_star = [lm_star]
	end

	# telluric models with n basis stellar model
	if is_time_variable(om.tel) #&& !seeded
		if pairwise
			lm_tel = [copy(lm_tel) for i in 1:(n_comp_star+1)]
			if multithread
				# @threads for i in 1:n_comp_star
				ThreadsX.foreach(1:n_comp_star) do i
					if log_lm(lm_star[1])
						_flux_star = _eval_lm(view(lm_star[1].M, :, 2:(i+1)), view(lm_star[1].s, 2:(i+1), :), _eval_lm(view(lm_star[1].M, :, 1:1), view(lm_star[1].s, 1:1, :), lm_star[1].μ); log_lm=log_lm(lm_star[1]))
					else
						_flux_star = _eval_lm(view(lm_star[1].M, :, 1:(i+1)), view(lm_star[1].s, 1:(i+1), :), lm_star[1].μ; log_lm=log_lm(lm_star[1]))
					end
					_flux_tel, _vars_tel = _spectra_interp_gp_div_gp(om.tel.log_λ, d.flux, d.var, d.log_λ_obs, _flux_star, vars_star, star_log_λ_tel)
					lm_tel[i+1].μ[:] = make_template(_flux_tel, _vars_tel; min=μ_min, max=μ_max, use_mean=true)
					mask_low_pixels!(_flux_tel, _vars_tel)
					mask_high_pixels!(_flux_tel, _vars_tel)
					EMPCA!(lm_tel[i+1].M, lm_tel[i+1].s, lm_tel[i+1].μ, _flux_tel, 1 ./ _vars_tel; log_lm=log_lm(lm_tel[i+1]))
				end
			else
				for i in 1:n_comp_star
					if log_lm(lm_star[1])
						flux_star .= _eval_lm(view(lm_star[1].M, :, 2:(i+1)), view(lm_star[1].s, 2:(i+1), :), _eval_lm(view(lm_star[1].M, :, 1:1), view(lm_star[1].s, 1:1, :), lm_star[1].μ); log_lm=log_lm(lm_star[1]))
					else
						flux_star .= _eval_lm(view(lm_star[1].M, :, 1:(i+1)), view(lm_star[1].s, 1:(i+1), :), lm_star[1].μ; log_lm=log_lm(lm_star[1]))
					end
					update_tel!(lm_star[1], lm_tel[i+1], true)
				end
			end
		else
			lm_tel = [copy(lm_tel) for i in 1:2]
			if log_lm(lm_star[1])
				flux_star .= _eval_lm(view(lm_star[1].M, :, 2:(n_comp_star+1)), view(lm_star[1].s, 2:(n_comp_star+1), :), _eval_lm(view(lm_star[1].M, :, 1:1), view(lm_star[1].s, 1:1, :), lm_star[1].μ); log_lm=log_lm(lm_star[1]))
			else
				flux_star .= _eval_lm(view(lm_star[1].M, :, 1:(n_comp_star+1)), view(lm_star[1].s, 1:(n_comp_star+1), :), lm_star[1].μ; log_lm=log_lm(lm_star[1]))
			end
			update_tel!(lm_star[1], lm_tel[2], true)
		end
	else
		lm_tel = [lm_tel]
	end

	remove_lm_score_means!(lm_tel)
	remove_lm_score_means!(lm_star)
	flip_basis_vectors!(lm_tel)
	flip_basis_vectors!(lm_star)

	fill_TelModel!(om, lm_tel[1])
	fill_StarModel!(om, lm_star[1])
	om.metadata[:todo][:initialized] = true

	return lm_tel, lm_star, stellar_dominated
end

function remove_lm_score_means!(lm::FullLinearModel; prop::Real=0.)
	if prop != 0.
		mean_s = Array{Float64}(undef, size(lm.s, 1), 1)
		for i in 1:size(lm.s, 1)
			mean_s[i, 1] = mean(winsor(view(lm.s, i, :); prop=prop))
		end
	else
		mean_s = mean(lm.s; dims=2)
	end
	if lm.log
		lm.μ .*= exp.(lm.M * mean_s)
	else
		lm.μ .+= lm.M * mean_s
	end
	lm.s .-= mean_s
end
remove_lm_score_means!(lm::LinearModel; kwargs...) = nothing
function remove_lm_score_means!(om::OrderModel; kwargs...)
	remove_lm_score_means!(om.star.lm; kwargs...)
	remove_lm_score_means!(om.tel.lm; kwargs...)
end
function remove_lm_score_means!(lms::Vector{<:LinearModel}; kwargs...)
	for lm in lms
		remove_lm_score_means!(lm; kwargs...)
	end
end

function flip_basis_vectors!(lm::FullLinearModel)
	flipper = -sign.(mean(lm.M; dims=1))  # make basis vector be absorption features
	lm.M .*= flipper
	lm.s .*= flipper'
end
flip_basis_vectors!(lm::LinearModel) = nothing
function flip_basis_vectors!(om::OrderModel)
	flip_basis_vectors!(om.star.lm)
	flip_basis_vectors!(om.tel.lm)
end
function flip_basis_vectors!(lms::Vector{<:LinearModel})
	for lm in lms
		flip_basis_vectors!(lm)
	end
end


fill_TelModel!(om::OrderModel, lm::LinearModel) =
	copy_to_LinearModel!(om.tel.lm, lm)
fill_TelModel!(om::OrderModel, lm::LinearModel, inds) =
	copy_to_LinearModel!(om.tel.lm, lm, inds)

function fill_StarModel!(om::OrderModel, lm::FullLinearModel; inds=2:size(lm.M, 2))
	if length(inds) > 0; @assert inds[1] > 1 end
	copy_to_LinearModel!(om.star.lm, lm, inds)
	if typeof(om) <: OrderModelDPCA
		om.rv.lm.M .= view(lm.M, :, 1)
		om.rv.lm.s[:] = view(lm.s, 1, :)
	else
		om.rv .= view(lm.s, 1, :) .* -light_speed_nu #TODO check if this is correct
	end
end

L1(a) = sum(abs, a)
L2(a) = sum(abs2, a)
L∞(Δ::VecOrMat{<:Real}) = maximum(Δ)
L∞(Δ) = maximum(L∞.(Δ))
function shared_attention(M)
	shared_attentions = M' * M
	return sum(shared_attentions) - sum(diag(shared_attentions))
end


function model_prior(lm, om::OrderModel, key::Symbol)
	reg = getfield(om, Symbol(:reg_, key))
	sm = getfield(om, key)
	isFullLinearModel = length(lm) > 2
	val = 0.

	if haskey(reg, :GP_μ) || haskey(reg, :L2_μ) || haskey(reg, :L1_μ) || haskey(reg, :L1_μ₊_factor)
		μ_mod = lm[1+2*isFullLinearModel] .- 1
		if haskey(reg, :L2_μ); val += L2(μ_mod) * reg[:L2_μ] end
		if haskey(reg, :L1_μ)
			val += L1(μ_mod) * reg[:L1_μ]
			# For some reason dot() works but BLAS.dot() doesn't
			if haskey(reg, :L1_μ₊_factor); val += dot(μ_mod, μ_mod .> 0) * reg[:L1_μ₊_factor] * reg[:L1_μ] end
		end
		# if haskey(reg, :GP_μ); val -= logpdf(SOAP_gp(getfield(om, key).log_λ), μ_mod) * reg[:GP_μ] end
		# if haskey(reg, :GP_μ); val -= gp_ℓ_nabla(μ_mod, sm.A_sde, sm.Σ_sde) * reg[:GP_μ] end
		if haskey(reg, :GP_μ); val -= gp_ℓ_precalc(sm.Δℓ_coeff, μ_mod, sm.A_sde, sm.Σ_sde) * reg[:GP_μ] end
	end
	if isFullLinearModel
		if haskey(reg, :shared_M); val += shared_attention(lm[1]) * reg[:shared_M] end
		if haskey(reg, :L2_M); val += L2(lm[1]) * reg[:L2_M] end
		if haskey(reg, :L1_M); val += L1(lm[1]) * reg[:L1_M] end
		# if haskey(reg, :GP_μ); val -= gp_ℓ_precalc(sm.Δℓ_coeff, view(lm[1], :, 1), sm.A_sde, sm.Σ_sde) * reg[:GP_μ] end
		if haskey(reg, :GP_M)
			for i in 1:size(lm[1], 2)
				val -= gp_ℓ_precalc(sm.Δℓ_coeff, lm[1][:, i], sm.A_sde, sm.Σ_sde) * reg[:GP_M]
			end
		end
		val += model_s_prior(lm[2], reg)
	end
	return val
end
model_prior(lm::Union{FullLinearModel, TemplateModel}, om::OrderModel, key::Symbol) = model_prior(vec(lm), om, key)

nonzero_key(reg, key) = haskey(reg, key) && reg[key] != 0
function model_s_prior(s, reg::Dict)
	if (nonzero_key(reg, :L1_M) || nonzero_key(reg, :L2_M) || nonzero_key(reg, :GP_M))
		return L2(s)
	end
	return 0
end

tel_prior(om::OrderModel) = tel_prior(om.tel.lm, om)
tel_prior(lm, om::OrderModel) = model_prior(lm, om, :tel)
star_prior(om::OrderModel) = star_prior(om.star.lm, om)
star_prior(lm, om::OrderModel) = model_prior(lm, om, :star)

total_model(tel, star, rv) = tel .* (star .+ rv)
total_model(tel, star) = tel .* star

struct OutputDPCA{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}} <: Output
	tel::AM
	star::AM
	rv::AM
	total::M
	function OutputDPCA(tel::AM, star::AM, rv::AM, total::M) where {T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}}
		@assert size(tel) == size(star) == size(rv) == size(total)
		new{T, AM, M}(tel, star, rv, total)
	end
end
function Output(om::OrderModelDPCA, d::Data)
	@assert size(om.b2o[1], 1) == size(d.flux, 1)
	return OutputDPCA(tel_model(om), star_model(om), rv_model(om), d)
end
OutputDPCA(tel, star, rv, d::GenericData) =
	OutputDPCA(tel, star, rv, total_model(tel, star, rv))
OutputDPCA(tel, star, rv, d::LSFData) =
	OutputDPCA(tel, star, rv, d.lsf * total_model(tel, star, rv))
Base.copy(o::OutputDPCA) = OutputDPCA(copy(tel), copy(star), copy(rv))
function recalc_total!(o::OutputDPCA, d::GenericData)
	o.total .= total_model(o.tel, o.star, o.rv)
end
function recalc_total!(o::OutputDPCA, d::LSFData)
	o.total .= d.lsf * total_model(o.tel, o.star, o.rv)
end
function Output!(o::OutputDPCA, om::OrderModelDPCA, d::Data)
	o.tel .= tel_model(om)
	o.star .= star_model(om)
	o.rv .= rv_model(om)
	recalc_total!(o, d)
end
struct OutputWobble{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}} <: Output
	tel::AM
	star::AM
	total::M
	function OutputWobble(tel::AM, star::AM, total::M) where {T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}}
		@assert size(tel) == size(star) == size(total)
		new{T, AM, M}(tel, star, total)
	end
end
function Output(om::OrderModelWobble, d::Data)
	@assert size(om.t2o[1], 1) == size(d.flux, 1)
	return OutputWobble(tel_model(om), star_model(om), d)
end
OutputWobble(tel, star, d::GenericData) =
	OutputWobble(tel, star, total_model(tel, star))
OutputWobble(tel, star, d::LSFData) =
	OutputWobble(tel, star, d.lsf * total_model(tel, star))
Base.copy(o::OutputWobble) = OutputWobble(copy(tel), copy(star))
function recalc_total!(o::OutputWobble, d::GenericData)
	o.total .= total_model(o.tel, o.star)
end
function recalc_total!(o::OutputWobble, d::LSFData)
	o.total .= d.lsf * total_model(o.tel, o.star)
end
function Output!(o::OutputWobble, om::OrderModelWobble, d::Data)
	o.tel .= tel_model(om)
	o.star .= star_model(om)
	recalc_total!(o, d)
end

function copy_reg!(from::OrderModel, to::OrderModel)
	copy_dict!(to.reg_tel, from.reg_tel)
	copy_dict!(to.reg_star, from.reg_star)
end
