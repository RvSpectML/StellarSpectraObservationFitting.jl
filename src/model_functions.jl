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
# using ThreadsX
import ExpectationMaximizationPCA as EMPCA

abstract type OrderModel end
abstract type Output end
abstract type Data end


"""
	D_to_rv(D)

Approximately converting a Doppler shift `D ≡ log(λ1/λ0)` in log-wavelength space to an RV using `(λ1-λ0)/λ0 = λ1/λ0 - 1 = e^D - 1 ≈ β = v / c` (with an extra negative sign from somewhere)
"""
D_to_rv(D) = light_speed_nu .* (1-exp.(D))
# function D_to_rv(D)
# 	x = exp.(2 .* D)
# 	return light_speed_nu .* ((1 .- x) ./ (1 .+ x))
# end

"""
	rv_to_D(v)

Approximately converting an RV to a Doppler shift `D ≡ log(λ1/λ0)` in log-wavelength space using `(λ1-λ0)/λ0 = λ1/λ0 - 1 = e^D - 1 ≈ β = v / c` (with an extra negative sign from somewhere)
"""
rv_to_D(v) = log1p.(-v ./ light_speed_nu)  # a simple approximation that minimally confuses that we are measuring redshift
# rv_to_D(v) = (log1p.((-v ./ light_speed_nu)) - log1p.((v ./ light_speed_nu))) ./ 2
# rv_to_D(v) = log.((1 .- v ./ light_speed_nu) ./ (1 .+ v ./ light_speed_nu)) ./ 2


"""
	_lower_inds!(lower_inds, lower_inds_adj, model_log_λ, rvs, log_λ_obs)

find the `model_log_λ` indices (incl. RV shifts) that bracket each `log_λ_obs`
"""
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


"""
	StellarInterpolationHelper

A linear interpolation holding information to interpolate the stellar model shifted by an RV to the data
"""
struct StellarInterpolationHelper{T1<:Real, T2<:Int}
	"`log_λ_obs - model_log_λ[lower_inds]` precomputed for coefficient calculations"
    log_λ_obs_m_model_log_λ_lo::AbstractMatrix{T1}
	"The stellar model log λ uniform step size"
	model_log_λ_step::T1
	"The lower indices"
	lower_inds::AbstractMatrix{T2}
	"The lower indices plus 1"
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
	for i in eachindex(inds)
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
Base.copy(s::StellarInterpolationHelper) = StellarInterpolationHelper(copy(s.log_λ_obs_m_model_log_λ_lo), s.model_log_λ_step, copy(s.lower_inds), copy(s.lower_inds_p1))


"""
	spectra_interp(model_flux, rvs, sih)

Interpolate the stellar model to the data using the lower inds in `sih`
"""
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
function spectra_interp(model_flux::AbstractVector, rv::Real, sih::StellarInterpolationHelper; sih_ind::Int=1)
	ratios = (view(sih.log_λ_obs_m_model_log_λ_lo, :, sih_ind) .+ rv_to_D(rv)) ./ sih.model_log_λ_step
	return (view(model_flux, view(sih.lower_inds, :, sih_ind)) .* (1 .- ratios)) + (view(model_flux, view(sih.lower_inds_p1, :, sih_ind)) .* ratios)
end
spectra_interp(model_flux, rvs, sih::StellarInterpolationHelper) =
	spectra_interp_nabla(model_flux, rvs, sih)
@explicit_intercepts spectra_interp Tuple{AbstractMatrix, AbstractVector, StellarInterpolationHelper} [true, false, false]
function Nabla.∇(::typeof(spectra_interp), ::Type{Arg{1}}, _, y, ȳ, model_flux, rvs, sih)
	ratios = (sih.log_λ_obs_m_model_log_λ_lo .+ rv_to_D(rvs)') ./ sih.model_log_λ_step
	ȳnew = zeros(size(model_flux, 1), size(ȳ, 2))
	# samp is λ_obs x λ_model
	for k in axes(ȳ, 1)  # λ_obs
		for j in axes(ȳnew, 2)  # time
			λ_model_lo = sih.lower_inds[k, j] - size(ȳnew, 1)*(j-1)
			# for i in axes(ȳnew, 1)  # λ_model
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


allequal(x) = all(y->y==x[1],x)

"""
	LSFData

Holds preprocessed data used to optimize SSOF models (including a matrix that approximates convolution with the instrument line spread function)
"""
struct LSFData{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{<:Number}} <: Data
    "Observed normalized flux"
	flux::AM
	"Observed normalized variance"
    var::AM
	"Observed normalized variance (with constant masking in barycentric frame)"
	var_s::AM
	"Log of observed wavelengths"
    log_λ_obs::AM
	"Pixel boundaries of the observed log wavelengths"
	log_λ_obs_bounds::M
	"Log of observed wavelengths (after barycentric correction)"
    log_λ_star::AM
	"Pixel boundaries of the observed log wavelengths (after barycentric correction)"
	log_λ_star_bounds::M
	"Matrix that approximates convolution with the instrument line spread function"
	lsf::Union{Vector{<:SparseMatrixCSC},SparseMatrixCSC}
	function LSFData(flux::AM, var::AM, var_s::AM, log_λ_obs::AM, log_λ_obs_bounds::M, log_λ_star::AM, log_λ_star_bounds::M, lsf::Union{Vector{<:SparseMatrixCSC},SparseMatrixCSC}) where {T<:Real, AM<:AbstractMatrix{T}, M<:Matrix{<:Number}}
		@assert size(flux) == size(var) == size(var_s) == size(log_λ_obs) == size(log_λ_star)
		if typeof(lsf) <: Vector
			@assert size(lsf[1], 1) == size(lsf[1], 2) == size(flux, 1)
			@assert allequal([size(l, 1) for l in lsf])
			@assert allequal([size(l, 2) for l in lsf])
		else
			@assert size(lsf, 1) == size(lsf, 2) == size(flux, 1)
		end
		return new{T, AM, M}(flux::AM, var::AM, var_s::AM, log_λ_obs::AM, log_λ_obs_bounds::M, log_λ_star::AM, log_λ_star_bounds::M, lsf)
	end
end
function LSFData(flux::AM, var::AM, var_s::AM, log_λ_obs::AM, log_λ_star::AM, lsf::Union{Vector{<:SparseMatrixCSC},SparseMatrixCSC}) where {T<:Real, AM<:AbstractMatrix{T}}
	log_λ_obs_bounds = bounds_generator(log_λ_obs)
	log_λ_star_bounds = bounds_generator(log_λ_star)
	return LSFData(flux, var, var_s, log_λ_obs, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds, lsf)
end
LSFData(flux::AM, var::AM, var_s::AM, log_λ_obs::AM, log_λ_star::AM, lsf::Nothing) where {T<:Real, AM<:AbstractMatrix{T}} =
	GenericData(flux, var, var_s, log_λ_obs, log_λ_star)
(d::LSFData)(inds::AbstractVecOrMat) =
	LSFData(view(d.flux, :, inds), view(d.var, :, inds), view(d.var_s, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds), d.lsf)
Base.copy(d::LSFData) = LSFData(copy(d.flux), copy(d.var), copy(d.var_s), copy(d.log_λ_obs), copy(d.log_λ_obs_bounds), copy(d.log_λ_star), copy(d.log_λ_star_bounds), copy(d.lsf))


"""
	GenericData

Holds preprocessed data used to optimize SSOF models
"""
struct GenericData{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{<:Number}} <: Data
    "Observed normalized flux"
	flux::AM
	"Observed normalized variance"
    var::AM
	"Observed normalized variance (with constant masking in barycentric frame)"
	var_s::AM
	"Log of observed wavelengths"
    log_λ_obs::AM
	"Pixel boundaries of the observed log wavelengths"
	log_λ_obs_bounds::M
	"Log of observed wavelengths (after barycentric correction)"
    log_λ_star::AM
	"Pixel boundaries of the observed log wavelengths (after barycentric correction)"
	log_λ_star_bounds::M
	function GenericData(flux::AM, var::AM, var_s::AM, log_λ_obs::AM, log_λ_star::AM) where {T<:Number, AM<:AbstractMatrix{T}}
		@assert size(flux) == size(var) == size(var_s) == size(log_λ_obs) == size(log_λ_star)
		log_λ_obs_bounds = bounds_generator(log_λ_obs)
		log_λ_star_bounds = bounds_generator(log_λ_star)
		return new{T, AM, typeof(log_λ_obs_bounds)}(flux, var, var_s, log_λ_obs, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds)
	end
end
(d::GenericData)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, :, inds), view(d.var, :, inds), view(d.var_s, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds))
Base.copy(d::GenericData) = GenericData(copy(d.flux), copy(d.var), copy(d.var_s), copy(d.log_λ_obs), copy(d.log_λ_star))
GenericData(d::LSFData) = GenericData(d.flux, d.var, d.var_s, d.log_λ_obs, d.log_λ_star)
GenericData(d::GenericData) = d


"""
	GenericDatum

Holds a single preprocessed spectrum used to optimize SSOF models
"""
struct GenericDatum{T<:Number, AV<:AbstractVector{T}, V<:Vector{<:Number}} <: Data
    "Observed normalized flux"
    flux::AV
    "Observed normalized variance"
    var::AV
	"Observed normalized variance (with constant masking in barycentric frame)"
	var_s::AV
    "Log of observed wavelengths"
    log_λ_obs::AV
	"Pixel boundaries of the observed log wavelengths"
	log_λ_obs_bounds::V
    "Log of observed wavelengths (after barycentric correction)"
    log_λ_star::AV
	"Pixel boundaries of the observed log wavelengths (after barycentric correction)"
	log_λ_star_bounds::V
	function GenericDatum(flux::AV, var::AV, log_λ_obs::AV, log_λ_star::AV) where {T<:Number, AV<:AbstractVector{T}}
		@assert size(flux) == size(var) == size(var_s) == size(log_λ_obs) == size(log_λ_star)
		log_λ_obs_bounds = bounds_generator(log_λ_obs)
		log_λ_star_bounds = bounds_generator(log_λ_star)
		return new{T, AV, typeof(log_λ_obs_bounds)}(flux, var, var_s, log_λ_obs, log_λ_obs_bounds, log_λ_star, log_λ_star_bounds)
	end
end
(d::GenericDatum)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, inds), view(d.var, inds), view(d.var_s, inds),
	view(d.log_λ_obs, inds), view(d.log_λ_star, inds))
Base.copy(d::GenericDatum) = GenericDatum(copy(d.flux), copy(d.var), copy(d.var_s), copy(d.log_λ_obs), copy(d.log_λ_star))
# combines many GenericDatum into a GenericData object
function GenericData(d::Vector{<:GenericDatum})
	len_obs = length(d[1].flux)
	n_obs = length(d)
	flux_obs = ones(len_obs, n_obs)
	var_obs = Array{Float64}(undef, len_obs, n_obs)
	var_obs_s = Array{Float64}(undef, len_obs, n_obs)
	log_λ_obs = Array{Float64}(undef, len_obs, n_obs)
	# log_λ_obs_bounds = Array{Float64}(undef, len_obs+1, n_obs)
	log_λ_star = Array{Float64}(undef, len_obs, n_obs)
	# log_λ_star_bounds = Array{Float64}(undef, len_obs+1, n_obs)
	for i in 1:n_obs # 13s
		flux_obs[:, i] .= d[i].flux
		var_obs[:, i] .= d[i].var
		var_obs_s[:, i] .= d[i].var_s
		log_λ_obs[:, i] .= d[i].log_λ_obs
		# log_λ_obs_bounds[:, i] .= d[i].log_λ_obs_bounds
		log_λ_star[:, i] .= d[i].log_λ_star
		# log_λ_star_bounds[:, i] .= d[i].log_λ_star_bounds
	end
	return GenericData(flux_obs, var_obs, var_obs_s, log_λ_obs, log_λ_star)
end


"""
	create_λ_template(log_λ_obs; upscale=1.)

Creating a uniform grid of log wavelengths for the SSOF models to be evaluated on
"""
function create_λ_template(log_λ_obs::AbstractMatrix; upscale::Real=1.)
    log_min_wav, log_max_wav = extrema(log_λ_obs)
	Δ_logλ_og = minimum(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # minimum pixel separation
	# Δ_logλ_og = minimum(view(log_λ_obs, size(log_λ_obs, 1), :) .- view(log_λ_obs, 1, :)) / size(log_λ_obs, 1)  # minimum avg pixel separation
	# Δ_logλ_og = median(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # median pixel separation
	# Δ_logλ_og = maximum(view(log_λ_obs, 2:size(log_λ_obs, 1), :) .- view(log_λ_obs, 1:size(log_λ_obs, 1)-1, :))  # maximum pixel separation
	Δ_logλ = Δ_logλ_og / upscale
    log_λ_template = (log_min_wav - 2 * Δ_logλ_og):Δ_logλ:(log_max_wav + 2 * Δ_logλ_og)
    λ_template = exp.(log_λ_template)
    return log_λ_template, λ_template
end

abstract type LinearModel end
_log_lm_default = true

"""
	FullLinearModel

A (log) linear model including a mean (either μ+M*s or μ*exp(M*s))
"""
struct FullLinearModel{T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}, AV<:AbstractVector{T}}  <: LinearModel
	"Feature vectors/weights that control where the model varies"
	M::AM1
	"Scores that control how much the model varies at each time"
	s::AM2
	"Template which controls the average behavior"
	μ::AV
	"Whether or not the model is log linear"
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


"""
	BaseLinearModel

A (log) linear model without a mean (either M*s or exp(M*s)).
Used for DPCA models
"""
struct BaseLinearModel{T<:Number, AM1<:AbstractMatrix{T}, AM2<:AbstractMatrix{T}} <: LinearModel
	"Feature vectors/weights that control where the model varies"
	M::AM1
	"Scores that control how much the model varies at each time"
	s::AM2
	"Whether or not the model is log linear"
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

"""
	TemplateModel

A constant model (when there are no feature vectors)
"""
struct TemplateModel{T<:Number, AV<:AbstractVector{T}} <: LinearModel
	μ::AV
	n::Int
end
Base.copy(tlm::TemplateModel) = TemplateModel(copy(tlm.μ), tlm.n)
LinearModel(tm::TemplateModel, inds::AbstractVecOrMat) = TemplateModel(tm.μ, length(inds))

"Whether a LinearModel is log linear"
log_lm(lm::TemplateModel) = false
log_lm(lm::LinearModel) = lm.log

# setting some syntactic sugar
Base.getindex(lm::LinearModel, s::Symbol) = getfield(lm, s)
Base.eachindex(lm::TemplateModel) = fieldnames(typeof(lm))
Base.eachindex(lm::LinearModel) = fieldnames(typeof(lm))[1:end-1]  # dealing with log fieldname
Base.setindex!(lm::LinearModel, a::AbstractVecOrMat, s::Symbol) = (lm[s] .= a)

# getting a vector of the linear model components
vec(lm::LinearModel) = [lm[i] for i in eachindex(lm)]
vec(lm::TemplateModel) = [lm.μ]
vec(lms::Vector{<:LinearModel}) = [vec(lm) for lm in lms]

LinearModel(M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector; log_lm::Bool=_log_lm_default) = FullLinearModel(M, s, μ, log_lm)
LinearModel(M::AbstractMatrix, s::AbstractMatrix; log_lm::Bool=_log_lm_default) = BaseLinearModel(M, s, log_lm)
LinearModel(μ::AbstractVector, n::Int) = TemplateModel(μ, n)

LinearModel(lm::FullLinearModel, s::AbstractMatrix) = FullLinearModel(lm.M, s, lm.μ, log)
LinearModel(lm::BaseLinearModel, s::AbstractMatrix) = BaseLinearModel(lm.M, s, lm.log)
LinearModel(lm::TemplateModel, s::AbstractMatrix) = lm

# Ref(lm::FullLinearModel) = [Ref(lm.M), Ref(lm.s), Ref(lm.μ)]
# Ref(lm::BaseLinearModel) = [Ref(lm.M), Ref(lm.s)]
# Ref(lm::TemplateModel) = [Ref(lm.μ)]


"""
	_eval_lm(M, s, μ; log_lm=false)

Evaluate a LinearModel
"""
_eval_lm(M, s, μ; log_lm::Bool=false) = log_lm ? (return exp.(M * s) .* μ) : (return (M * s) .+ μ)
_eval_lm(M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector) = muladd(M, s, μ)  # faster, but Nabla doesn't handle it
_eval_lm(M, s; log_lm::Bool=false) = log_lm ? (return exp.(M * s)) : (return (M * s))
# _eval_lm(μ, n::Int) = repeat(μ, 1, n)
_eval_lm(μ, n::Int) = μ * ones(n)'  # this is faster I dont know why
_eval_lm(flm::FullLinearModel) = _eval_lm(flm.M, flm.s, flm.μ; log_lm=flm.log)
_eval_lm(blm::BaseLinearModel) = _eval_lm(blm.M, blm.s; log_lm=blm.log)
_eval_lm(tlm::TemplateModel) = _eval_lm(tlm.μ, tlm.n)
(lm::LinearModel)() = _eval_lm(lm)

(flm::FullLinearModel)(inds::AbstractVecOrMat) = _eval_lm(view(flm.M, inds, :), flm.s, flm.μ; log_lm=flm.log)
(blm::BaseLinearModel)(inds::AbstractVecOrMat) = _eval_lm(view(blm.M, inds, :), blm.s; log_lm=blm.log)
(tlm::TemplateModel)(inds::AbstractVecOrMat) = repeat(view(tlm.μ, inds), 1, tlm.n)

function copy_TemplateModel!(to::LinearModel, from::LinearModel)
	to.μ .= from.μ
end
copy_to_LinearModel!(to::TemplateModel, from::TemplateModel) = copy_TemplateModel!(to, from)
copy_to_LinearModel!(to::LinearModel, from::TemplateModel) = copy_TemplateModel!(to, from)
copy_to_LinearModel!(to::TemplateModel, from::LinearModel) = copy_TemplateModel!(to, from)
function copy_to_LinearModel!(to::LinearModel, from::LinearModel)
	@assert typeof(to) == typeof(from)
	@assert to.log == from.log
	for i in eachindex(from)
		getfield(to, i) .= getfield(from, i)
	end
end
copy_to_LinearModel!(to::TemplateModel, from::LinearModel, inds) =
	copy_to_LinearModel!(to, from)
copy_to_LinearModel!(to::LinearModel, from::TemplateModel, inds) =
	copy_to_LinearModel!(to, from)
copy_to_LinearModel!(to::TemplateModel, from::TemplateModel, inds) =
	copy_to_LinearModel!(to, from)
nvec(lm::TemplateModel) = 0
nvec(lm::LinearModel) = size(lm.M, 2)
function copy_to_LinearModel!(to::FullLinearModel, from::FullLinearModel, inds)
	@assert to.log == from.log
	to.μ .= from.μ
	to.M[:, inds] .= view(from.M, :, inds)
	to.s[inds, :] .= view(from.s, inds, :)
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


"""
	Submodel

Holds information on the wavelengths, LTISDE representaiton for the GP reguarlization term, and linear model for a SSOF model component
"""
struct Submodel{T<:Number, AV1<:AbstractVector{T}, AV2<:AbstractVector{T}, AA<:AbstractArray{T}}
    "Uniform separation log wavelengths of the SSOF model component"
	log_λ::AV1
	"`exp.(log_λ)`"
    λ::AV2
	"Linear model"
	lm::LinearModel
	"State transition matrix"
	A_sde::StaticMatrix
	"Process noise"
	Σ_sde::StaticMatrix
	"Coefficients that can be used to calculate the gradient of the GP regularization term"
	Δℓ_coeff::AA
end
function Submodel(log_λ_obs::AbstractVecOrMat, n_comp::Int, log_λ_gp::Real; include_mean::Bool=true, log_lm::Bool=_log_lm_default, log_λ::Union{Nothing,AbstractRange}=nothing, kwargs...)
	n_obs = size(log_λ_obs, 2)
	if isnothing(log_λ)
		log_λ, λ = create_λ_template(log_λ_obs; kwargs...)
	else
		λ = exp.(log_λ)
	end
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


"""
	_shift_log_λ_model(log_λ_obs_from, log_λ_obs_to, log_λ_model_from)

Getting model log λ in a different reference frame using the shifts in the different `log_λ_obs`
"""
function _shift_log_λ_model(log_λ_obs_from, log_λ_obs_to, log_λ_model_from)
	n_obs = size(log_λ_obs_from, 2)
	dop = [log_λ_obs_from[1, i] - log_λ_obs_to[1, i] for i in 1:n_obs]
	log_λ_model_to = ones(length(log_λ_model_from), n_obs)
	for i in 1:n_obs
		log_λ_model_to[:, i] .= log_λ_model_from .+ dop[i]
	end
	return log_λ_model_to
end

# Default regularization values
# They need to be different or else the stellar μ will be surpressed
default_reg_tel = Dict([(:GP_μ, 1e6), (:L2_μ, 1e6), (:L1_μ, 1e5), (:L1_μ₊_factor, 6.),
	(:GP_M, 1e7), (:L1_M, 1e7)])
default_reg_star = Dict([(:GP_μ, 1e2), (:L2_μ, 1e-2), (:L1_μ, 1e2), (:L1_μ₊_factor, 6.),
	(:GP_M, 1e4), (:L1_M, 1e7)])
default_reg_model = Dict([(:L2_μ, 1e6), (:L2_h2o, 1e6), (:L2_oxy, 1e6)])
#default_reg_model = Dict([(:L2_μ, 1e10), (:L2_h2o, 1e10), (:L2_oxy, 1e10), (:GP_μ_star, 1e10)])
#default_reg_model = Dict([(:L2_μ, 1e10), (:GP_μ, 1e10)])
#default_reg_model = Dict([(:L2_μ, 1e10),])
#default_reg_model = Dict([(:L2_μ, 1e10), (:GP_μ, 1e10), (:GP_h2o, 1e10), (:GP_oxy, 1e10)])
#default_reg_model = Dict([(:L2_μ, 1e10), (:L2_h2o, 1e10), (:L2_oxy, 1e10), (:GP_μ, 1e10)])
#default_reg_model = Dict([(:L2_μ, 1e10), (:L2_h2o, 1e10), (:L2_oxy, 1e10), (:GP_μ, 1e10), (:GP_h2o, 1e10), (:GP_oxy, 1e10)])
#default_reg_model = Dict([(:L2_h2o, 1e10), (:L2_oxy, 1e10)])
default_reg_tel_full = Dict([(:GP_μ, 1e6), (:L2_μ, 1e6), (:L1_μ, 1e5),
	(:L1_μ₊_factor, 6.), (:GP_M, 1e7), (:L2_M, 1e4), (:L1_M, 1e7)])
default_reg_star_full = Dict([(:GP_μ, 1e2), (:L2_μ, 1e-2), (:L1_μ, 1e1),
	(:L1_μ₊_factor, 6.), (:GP_M, 1e4), (:L2_M, 1e4), (:L1_M, 1e7)])


"""
	oversamp_interp_helper(to_bounds, from_x)

Finding the coefficients to integrate the model between observed pixel bounds
"""
function oversamp_interp_helper(to_bounds::AbstractVector, from_x::AbstractVector)
	ans = spzeros(length(to_bounds)-1, length(from_x))
	bounds_inds = searchsortednearest(from_x, to_bounds)
	for i in axes(ans, 1)
		x_lo, x_hi = to_bounds[i], to_bounds[i+1]  # values of bounds
		lo_ind, hi_ind = bounds_inds[i], bounds_inds[i+1]  # indices of points in model closest to the bounds

		# if necessary, shrink so that so from_x[lo_ind] and from_x[hi_ind] are in the bounds
		if from_x[lo_ind] < x_lo; lo_ind += 1 end
		if from_x[hi_ind] > x_hi; hi_ind -= 1 end

		edge_term_lo = (from_x[lo_ind] - x_lo) ^ 2 / (from_x[lo_ind] - from_x[lo_ind-1])
		edge_term_hi = (x_hi - from_x[hi_ind]) ^ 2 / (from_x[hi_ind+1] - from_x[hi_ind])

		ans[i, lo_ind-1] = edge_term_lo
		ans[i, hi_ind+1] = edge_term_hi
		if lo_ind==hi_ind
			ans[i, lo_ind] = from_x[lo_ind+1] + from_x[lo_ind] - 2 * x_lo - edge_term_lo - edge_term_hi
		else
			ans[i, lo_ind] = from_x[lo_ind+1] + from_x[lo_ind] - 2 * x_lo - edge_term_lo
			ans[i, lo_ind+1:hi_ind-1] .= view(from_x, lo_ind+2:hi_ind) .- view(from_x, lo_ind:hi_ind-2)
			ans[i, hi_ind] = 2 * x_hi - from_x[hi_ind] - from_x[hi_ind-1] - edge_term_hi
		end
		# println(sum(view(ans, i, lo_ind-1:hi_ind+1))," vs ", 2 * (x_hi - x_lo))
		# @assert isapprox(sum(view(ans, i, lo_ind-1:hi_ind+1)), 2 * (x_hi - x_lo); rtol=1e-3)
		ans[i, lo_ind-1:hi_ind+1] ./= sum(view(ans, i, lo_ind-1:hi_ind+1))
		# ans[i, lo_ind-1:hi_ind+1] ./= 2 * (x_hi - x_lo)
	end
	dropzeros!(ans)
	return ans
end
oversamp_interp_helper(to_bounds::AbstractMatrix, from_x::AbstractVector) =
	[oversamp_interp_helper(view(to_bounds, :, i), from_x) for i in axes(to_bounds, 2)]
oversamp_interp_helper(to_x::AbstractVector, from_x::AbstractMatrix) =
	[oversamp_interp_helper(to_bounds, view(from_x, :, i)) for i in axes(from_x, 2)]


"""
	undersamp_interp_helper(to_x, from_x)

Finding the coefficients to linear interpolate the model at observed pixel locations
"""
function undersamp_interp_helper(to_x::AbstractVector, from_x::AbstractVector)
	ans = spzeros(length(to_x), length(from_x))
	# ans = sparse(Float64[],Float64[],Float64[],length(to_x),length(from_x))
	to_inds = searchsortednearest(from_x, to_x; lower=true)
	for i in axes(ans, 1)
		x_new = to_x[i]
		ind = to_inds[i]  # index of point in model below to_x[i]
		if ind < length(from_x)
			dif = (x_new-from_x[ind]) / (from_x[ind+1] - from_x[ind])
			ans[i, ind] = 1 - dif
			ans[i, ind + 1] = dif
		else
			ans[i, ind] = 1
		end
	end
	dropzeros!(ans)
	return ans
end
undersamp_interp_helper(to_x::AbstractMatrix, from_x::AbstractVector) =
	[undersamp_interp_helper(view(to_x, :, i), from_x) for i in axes(to_x, 2)]
undersamp_interp_helper(to_x::AbstractVector, from_x::AbstractMatrix) =
	[undersamp_interp_helper(to_x, view(from_x, :, i)) for i in axes(from_x, 2)]


"""
	OrderModelDPCA

SSOF model for a set of 1D spectra using Doppler-constrained PCA to measure the RVs (which are contained in the RV Submodel)
"""
struct OrderModelDPCA{T<:Number} <: OrderModel
	"Telluric submodel"
    tel::Submodel
	"Stellar submodel"
    star::Submodel
	"RV submodel with Doppler feature vector"
	rv::Submodel
	"Telluric regularization coefficients"
	reg_tel::Dict{Symbol, T}
	"Stellar regularization coefficients"
	reg_star::Dict{Symbol, T}
	"Telluric model regularization coefficients"
	reg_model::Dict{Symbol, T}
	"Matrices to interpolate stellar model to observed wavelengths (barycenter 2 observed)"
	b2o::AbstractVector{<:SparseMatrixCSC}
	"Matrices to interpolate telluric model to observed wavelengths (telluric 2 observed)"
	t2o::AbstractVector{<:SparseMatrixCSC}
	"Holds the data on the instrument, star, and order of data being as well as whether the model has been optimized or regularized etc."
	metadata::Dict{Symbol, Any}
	"Number of spectra to be modeled"
	n::Int
end

"""
	OrderModelWobble

SSOF model for a set of 1D spectra using linear interpolation of the stellar model to measure the RVs
"""
struct OrderModelWobble{T<:Number} <: OrderModel
	"Telluric submodel"
    tel::Submodel
	"Stellar submodel"
    star::Submodel
	"RVs"
	rv::AbstractVector
	"Telluric regularization coefficients"
	reg_tel::Dict{Symbol, T}
	"Stellar regularization coefficients"
	reg_star::Dict{Symbol, T}
	"Telluric model regularization coefficients"
	reg_model::Dict{Symbol, T}
	"Linear interpolation helper objects to interpolate stellar model to observed wavelengths (barycenter 2 observed)"
	b2o::StellarInterpolationHelper
	"Approximate barycentric correction \"RVs\" (using `(λ1-λ0)/λ0 = λ1/λ0 - 1 = e^D - 1 ≈ β = v / c`)"
	bary_rvs::AbstractVector{<:Real}
	"Matrices to interpolate telluric model to observed wavelengths (telluric 2 observed)"
	t2o::AbstractVector{<:SparseMatrixCSC}
	"Holds the data on the instrument, star, and order of data being as well as whether the model has been optimized or regularized etc."
	metadata::Dict{Symbol, Any}
	"Number of spectra to be modeled"
	n::Int
end


"""
	OrderModel(d; kwargs...)
	
Constructor for the `OrderModel`-type objects (SSOF model for a set of 1D spectra)
	
# Optional arguments
- `instrument::String="None"`: The name of the instrument(s) the data was taken from. For bookkeeping
- `order::Int=0`: What order (if any) the data was taken from. For bookkeeping
- `star_str::String="None"`: The name of the star the data was taken from. For bookkeeping
- `n_comp_tel::Int=5`: Amount of telluric feature vectors
- `n_comp_star::Int=5`: The maximum amount of stellar feature vectors
- `oversamp::Bool=true`: Whether or not to integrate the model or do linear interpolation (for the telluric model)
- `dpca::Bool=false`: Whether to use Doppler-constrained PCA or variable interpolation location to determine the RVs
- `log_λ_gp_star::Real=1/SOAP_gp_params.λ`: The log λ lengthscale of the stellar regularization GP
- `log_λ_gp_tel::Real=1/LSF_gp_params.λ`: The log λ lengthscale of the telluric regularization GP
- `tel_log_λ::Union{Nothing,AbstractRange}=nothing`: The log
- `star_log_λ::Union{Nothing,AbstractRange}=nothing`: The log λ lengthscale of the telluric regularization GP
- `use_tel_prior::Bool=false`: Whether to include a modeled prior in the telluric submodel
- `reg_star::Union{Nothing,Dict{Symbol,Float64}}=nothing`: Overrides default_reg_star values if given
- `reg_tel::Union{Nothing,Dict{Symbol,Float64}}=nothing`: Overrides default_reg_tel values if given
- `reg_model::Union{Nothing,Dict{Symbol,Float64}}=nothing`: Overrides default_reg_model values if given
- `tel_below_one::Bool=false`: Whether to impose a heavy penalty for any telluric vector flux going above 1
- `fixed_vectors::Dict=Dict()`: Fix any stellar or telluric vectors. Must contain 'log_λ'
- `kwargs...`: kwargs passed to `Submodel` constructors
"""
function OrderModel(
	d::Data;
	instrument::String="None",
	order::Int=0,
	star_str::String="None",
	n_comp_tel::Int=2,
	n_comp_star::Int=2,
	oversamp::Bool=true,
	dpca::Bool=false,
	log_λ_gp_star::Real=1/SOAP_gp_params.λ,
	log_λ_gp_tel::Real=1/LSF_gp_params.λ,
	tel_log_λ::Union{Nothing,AbstractRange}=nothing,
	star_log_λ::Union{Nothing,AbstractRange}=nothing,
	use_tel_prior::Bool=false,
	reg_star::Union{Nothing,Dict{Symbol,Float64}}=nothing,
	reg_tel::Union{Nothing,Dict{Symbol,Float64}}=nothing,
	reg_model::Union{Nothing,Dict{Symbol,Float64}}=nothing,
	tel_below_one::Bool=false,
	fixed_vectors::Dict=Dict(),
	kwargs...)

	if isnothing(reg_star)
		reg_star = default_reg_star
	end
	if isnothing(reg_tel)
		reg_tel = default_reg_tel
	end
	if isnothing(reg_model)
		reg_model = default_reg_model
	end

	# Creating models
	tel = Submodel(d.log_λ_obs, n_comp_tel, log_λ_gp_tel; log_λ=tel_log_λ, kwargs...)
	star = Submodel(d.log_λ_star, n_comp_star, log_λ_gp_star; log_λ=star_log_λ, kwargs...)
	n_obs = size(d.log_λ_obs, 2)
	dpca ?
		rv = Submodel(d.log_λ_star, 1, log_λ_gp_star; include_mean=false, log_λ=star_log_λ, kwargs...) :
		rv = zeros(n_obs)

	bary_rvs = D_to_rv.([median(d.log_λ_star[:, i] - d.log_λ_obs[:, i]) for i in 1:n_obs])
	todo = Dict([(:initialized, false), (:reg_improved, false), (:err_estimated, false), (:h2o_tried, false), (:oxy_tried, false)])
	metadata = Dict([(:todo, todo), (:tel_prior, use_tel_prior), (:tel_below_one, tel_below_one), (:instrument, instrument), (:order, order), 
					 (:star, star_str), (:h2o_vector, 0), (:oxy_vector, 0), (:fixed_vectors, fixed_vectors)])
	# Pre-compute prior vectors
	if use_tel_prior
		metadata[:tel_prior_spec] = get_telluric_prior_spectrum(tel.λ, :all)
		metadata[:h2o_prior_spec] = get_telluric_prior_spectrum(tel.λ, :h2o)
		metadata[:oxy_prior_spec] = get_telluric_prior_spectrum(tel.λ, :oxy)
	end
	if dpca
		if oversamp
			b2o = oversamp_interp_helper(d.log_λ_star_bounds, star.log_λ)
			t2o = oversamp_interp_helper(d.log_λ_obs_bounds, tel.log_λ)
		else
			b2o = undersamp_interp_helper(d.log_λ_star, star.log_λ)
			t2o = undersamp_interp_helper(d.log_λ_obs, tel.log_λ)
		end
		return OrderModelDPCA(tel, star, rv, copy(reg_tel), copy(reg_star), copy(reg_model), b2o, t2o, metadata, n_obs)
	else
		b2o = StellarInterpolationHelper(star.log_λ, bary_rvs, d.log_λ_obs)
		if oversamp
			t2o = oversamp_interp_helper(d.log_λ_obs_bounds, tel.log_λ)
		else
			t2o = undersamp_interp_helper(d.log_λ_obs, tel.log_λ)
		end
		return OrderModelWobble(tel, star, rv, copy(reg_tel), copy(reg_star), copy(reg_model), b2o, bary_rvs, t2o, metadata, n_obs)
	end
end
Base.copy(om::OrderModelDPCA) = OrderModelDPCA(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), copy(om.reg_model), 
	om.b2o, om.t2o, deepcopy(om.metadata), om.n)
(om::OrderModelDPCA)(inds::AbstractVecOrMat) =
	OrderModelDPCA(om.tel(inds), om.star(inds), om.rv(inds), copy(om.reg_tel),
		copy(om.reg_star), copy(om.reg_model), view(om.b2o, inds), view(om.t2o, inds), deepcopy(om.metadata), length(inds))
Base.copy(om::OrderModelWobble) = OrderModelWobble(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), copy(om.reg_model),
	copy(om.b2o), om.bary_rvs, om.t2o, deepcopy(om.metadata), om.n)
(om::OrderModelWobble)(inds::AbstractVecOrMat) =
	OrderModelWobble(om.tel(inds), om.star(inds), view(om.rv, inds), copy(om.reg_tel), copy(om.reg_star), copy(om.reg_model), 
		om.b2o(inds, length(om.star.lm.μ)), view(om.bary_rvs, inds), view(om.t2o, inds), deepcopy(om.metadata), length(inds))


"""
	rm_dict!(d)

Remove all keys in a Dict
"""
function rm_dict!(d::Dict)
	for (key, value) in d
		delete!(d, key)
	end
end


"""
	rm_regularization!(om)

Remove all of the keys in the regularization Dicts
Currently does not affect om.reg_model
"""
function rm_regularization!(om::OrderModel)
	rm_dict!(om.reg_tel)
	rm_dict!(om.reg_star)
end


"""
	rm_dict!(d, key_start)

Remove all keys starting with `key_start` in a Dict
"""
function rm_dict!(d::Dict, key_start::String)
	for (key, value) in d
		if string(key)[1:length(key_start)] == key_start; delete!(d, key) end
	end
end


"""
	rm_GP_regularization!(om)

Remove all of the GP keys in the regularization Dicts
"""
function rm_GP_regularization!(om::OrderModel)
	rm_dict!(om.reg_tel, "GP_")
	rm_dict!(om.reg_star, "GP_")
end


"""
	zero_regularization(om; include_L1_factor=false)

Zero-out all of the keys in the regularization Dicts
"""
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


"""
	reset_regularization!(om)

Reset all of the keys in the regularization Dicts to the default values
"""
function reset_regularization!(om::OrderModel)
	for (key, value) in om.reg_tel
		om.reg_tel[key] = default_reg_tel_full[key]
	end
	for (key, value) in om.reg_star
		om.reg_star[key] = default_reg_star_full[key]
	end
end


"""
	_eval_lm_vec(om, v; log_lm=_log_lm_default)

Evaluate a vectorized version of a linear model in `v`
"""
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


"""
	rvs(model)

Get RVs (in m/s) from an OrderModel
The negative sign is only in the DPCA version because of how SSOF was originally coded
"""
rvs(model::OrderModelDPCA) = vec(model.rv.lm.s .* -light_speed_nu)
rvs(model::OrderModelWobble) = model.rv


"""
	downsize(lm, n_comp)

Create a smaller version of `lm` that only copies some amount of the feature vectors
"""
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
		copy(m.rv), copy(m.reg_tel), copy(m.reg_star), copy(m.reg_model), m.b2o, m.t2o, deepcopy(m.metadata), m.n)
downsize(m::OrderModelWobble, n_comp_tel::Int, n_comp_star::Int) =
	OrderModelWobble(
		downsize(m.tel, n_comp_tel),
		downsize(m.star, n_comp_star),
		copy(m.rv), copy(m.reg_tel), copy(m.reg_star), copy(m.reg_model), copy(m.b2o), m.bary_rvs, m.t2o, deepcopy(m.metadata), m.n)


"""
	downsize(lm, n_comp)

Create a smaller view of `lm` that can only see some amount of the feature vectors
"""
function downsize_view(lm::FullLinearModel, n_comp::Int)
	if n_comp > 0
		return FullLinearModel(view(lm.M, :, 1:n_comp), view(lm.s, 1:n_comp, :), lm.μ, lm.log)
	else
		return TemplateModel(lm.μ, size(lm.s, 2))
	end
end
downsize_view(lm::BaseLinearModel, n_comp::Int) =
	BaseLinearModel(view(lm.M, :, 1:n_comp), view(lm.s, 1:n_comp, :), lm.log)
function downsize_view(lm::TemplateModel, n_comp::Int)
	@assert n_comp==0
	return lm
end
downsize_view(sm::Submodel, n_comp::Int) =
	Submodel(sm.log_λ, sm.λ, downsize_view(sm.lm, n_comp), sm.A_sde, sm.Σ_sde, sm.Δℓ_coeff)
downsize_view(m::OrderModelDPCA, n_comp_tel::Int, n_comp_star::Int) =
	OrderModelDPCA(
		downsize_view(m.tel, n_comp_tel),
		downsize_view(m.star, n_comp_star),
		m.rv, copy(m.reg_tel), copy(m.reg_star), copy(m.reg_model), m.b2o, m.t2o, deepcopy(m.metadata), m.n)
downsize_view(m::OrderModelWobble, n_comp_tel::Int, n_comp_star::Int) =
	OrderModelWobble(
		downsize_view(m.tel, n_comp_tel),
		downsize_view(m.star, n_comp_star),
		m.rv, copy(m.reg_tel), copy(m.reg_star), copy(m.reg_model), m.b2o, m.bary_rvs, m.t2o, deepcopy(m.metadata), m.n)

"""
	spectra_interp(model, interp_helper)

Interpolates `model` using the interoplation described by `interp_helper`
"""
spectra_interp(model, interp_helper::SparseMatrixCSC) =
	interp_helper * model
spectra_interp(model::AbstractMatrix, interp_helper::AbstractVector{<:SparseMatrixCSC}) =
	hcat([spectra_interp(view(model, :, i), interp_helper[i]) for i in axes(model, 2)]...)
# spectra_interp_nabla(model, interp_helper::AbstractVector{<:SparseMatrixCSC}) =
# 	hcat([interp_helper[i] * model[:, i] for i in axes(model, 2)]...)
# spectra_interp(model, interp_helper::AbstractVector{<:SparseMatrixCSC}) =
# 	spectra_interp_nabla(model, interp_helper)
@explicit_intercepts spectra_interp Tuple{AbstractMatrix, AbstractVector{<:SparseMatrixCSC}} [true, false]
Nabla.∇(::typeof(spectra_interp), ::Type{Arg{1}}, _, y, ȳ, model, interp_helper) =
	hcat([interp_helper[i]' * view(ȳ, :, i) for i in axes(model, 2)]...)


"""
	tel_model(om; lm=om.tel.lm)

Gets telluric model interpolated onto the observed wavelengths
"""
tel_model(om::OrderModel; lm::LinearModel=om.tel.lm) = spectra_interp(lm(), om.t2o)


"""
	star_model(om; lm=om.star.lm)

Gets stellar model interpolated onto the observed wavelengths
"""
star_model(om::OrderModelDPCA; lm::LinearModel=om.star.lm) = spectra_interp(lm(), om.b2o)
star_model(om::OrderModelWobble; lm::LinearModel=om.star.lm) = spectra_interp(lm(), om.rv .+ om.bary_rvs, om.b2o)


"""
	rv_model(om; lm=om.rv.lm)

Gets RV model interpolated onto the observed wavelengths (used by OrderModelDPCA)
"""
rv_model(om::OrderModelDPCA; lm::LinearModel=om.rv.lm) = spectra_interp(lm(), om.b2o)


# function fix_FullLinearModel_s!(flm, min::Number, max::Number)
# 	@assert all(min .< flm.μ .< max)
# 	result = ones(typeof(flm.μ[1]), length(flm.μ))
# 	for i in axes(flm.s, 2)
# 		result[:] = _eval_lm(flm.M, flm.s[:, i], flm.μ)
# 		while any(result .> max) || any(result .< min)
# 			# println("$i, old s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
# 			flm.s[:, i] ./= 2
# 			result[:] = _eval_lm(flm.M, view(flm.s, :, i), flm.μ)
# 			# println("$i, new s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
# 		end
# 	end
# end


"""
	get_marginal_GP(finite_GP, ys, xs)

Marginalizes a TemporalGPs GP on `ys` and `xs`
"""
function get_marginal_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    gp_post = posterior(finite_GP, ys)
    gpx_post = gp_post(xs)
    return TemporalGPs.marginals(gpx_post)
end


"""
	get_mean_GP(finite_GP, ys, xs)

Gets the mean of the posterior of a TemporalGPs GP
"""
function get_mean_GP(
    finite_GP::Distribution{Multivariate,Continuous},
    ys::AbstractVector,
    xs::AbstractVector)
    return mean.(get_marginal_GP(finite_GP, ys, xs))
end


"""
	build_gp(params)

Builds a Matern 5/2 TemporalGPs GP
"""
function build_gp(params::NamedTuple)
	f_naive = AbstractGPs.GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
	return to_sde(f_naive, SArrayStorage(Float64))
end


# default TemporalGPs lengthscales fit to either SOAP data or a LSF line
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


"""
	_spectra_interp_gp!(fluxes, log_λ, flux_obs, var_obs, log_λ_obs; gp_mean=0., gp_base=SOAP_gp, mask_flux=Array{Bool}(undef, length(flux_obs)), mask_var=Array{Bool}(undef, length(flux_obs)))

Use a GP to interpolate `flux_obs` observed at `log_λ_obs` onto `log_λ`
"""
function _spectra_interp_gp!(fluxes::AbstractVector, log_λ, flux_obs::AbstractVector, var_obs, log_λ_obs; gp_mean::Number=0., gp_base=SOAP_gp, mask_flux::Vector=Array{Bool}(undef, length(flux_obs)), mask_var::Vector=Array{Bool}(undef, length(flux_obs)))
	mask_var[:] = isfinite.(var_obs)
	if !any(mask_var)
		var_obs[.!mask_var] .= 1e6*maximum(var_obs[mask_var])
	end
	mask_flux[:] = isfinite.(flux_obs)
	flux_obs[.!mask_flux] .= gp_mean
	gp = get_marginal_GP(gp_base(log_λ_obs, var_obs), flux_obs .- gp_mean, log_λ)
	fluxes[:] = mean.(gp) .+ gp_mean
	if !any(mask_var)
		var_obs[.!mask_var] .= Inf
	end
	flux_obs[.!mask_flux] .= Inf
	return gp
end
function _spectra_interp_gp!(fluxes::AbstractVector, vars, log_λ, flux_obs::AbstractVector, var_obs, log_λ_obs; keep_mask::Bool=true, kwargs...)
	gp = _spectra_interp_gp!(fluxes, log_λ, flux_obs, var_obs, log_λ_obs; kwargs...)
	vars[:] = var.(gp)
	if keep_mask
		inds = searchsortednearest(log_λ_obs, log_λ; lower=true)
		# for i in eachindex(inds)
		for i in eachindex(log_λ)
			if log_λ[i] <= log_λ_obs[1]
				if isinf(var_obs[1]); vars[i] = Inf end
			elseif log_λ[i] >= log_λ_obs[end]
				if isinf(var_obs[end]); vars[i] = Inf end
			elseif isinf(var_obs[inds[i]]) || isinf(var_obs[inds[i]+1])
				vars[i] = Inf
			end
		end
	end
	return gp
end
function _spectra_interp_gp!(fluxes::AbstractMatrix, log_λ, flux_obs, var_obs, log_λ_obs; var_kernel=SOAP_gp_params.var_kernel, λ_kernel=SOAP_gp_params.λ, kwargs...)
	gp_base = build_gp((var_kernel = var_kernel, λ = λ_kernel))
	for i in axes(flux_obs, 2)
		_spectra_interp_gp!(view(fluxes, :, i), log_λ, view(flux_obs, :, i), view(var_obs, :, i), view(log_λ_obs, :, i); mask_flux=Array{Bool}(undef, size(flux_obs, 1)), mask_var=Array{Bool}(undef, size(flux_obs, 1)),  gp_base=gp_base, kwargs...)
	end
end
function _spectra_interp_gp!(fluxes::AbstractMatrix, vars, log_λ, flux_obs, var_obs, log_λ_obs; var_kernel=SOAP_gp_params.var_kernel, λ_kernel=SOAP_gp_params.λ, kwargs...)
	gp_base = build_gp((var_kernel = var_kernel, λ = λ_kernel))
	for i in axes(flux_obs, 2)
		_spectra_interp_gp!(view(fluxes, :, i), view(vars, :, i), log_λ, view(flux_obs, :, i), view(var_obs, :, i), view(log_λ_obs, :, i); mask_flux=Array{Bool}(undef, size(flux_obs, 1)), mask_var=Array{Bool}(undef, size(flux_obs, 1)), gp_base=gp_base, kwargs...)
	end
end


"""
	remove_lm_score_means!(lm; prop=0.)

Recenter the scores in `lm` around 0
"""
function remove_lm_score_means!(lm::FullLinearModel; prop::Real=0.)
	if prop != 0.
		mean_s = Array{Float64}(undef, size(lm.s, 1), 1)
		for i in axes(lm.s, 1)
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


"""
	flip_feature_vectors!(lm)

Make each feature vector's median value negative
"""
function flip_feature_vectors!(lm::FullLinearModel)
	flipper = -sign.(mean(lm.M; dims=1) - median(lm.M; dims=1))  # make basis vector be absorption features
	lm.M .*= flipper
	lm.s .*= flipper'
end
flip_feature_vectors!(lm::LinearModel) = nothing
function flip_feature_vectors!(om::OrderModel)
	flip_feature_vectors!(om.star.lm)
	flip_feature_vectors!(om.tel.lm)
end
function flip_feature_vectors!(lms::Vector{<:LinearModel})
	for lm in lms
		flip_feature_vectors!(lm)
	end
end


"""
	fill_TelModel!(om, lm)

Replace the telluric model in `om` with `lm`
"""
fill_TelModel!(om::OrderModel, lm::LinearModel) =
	copy_to_LinearModel!(om.tel.lm, lm)
fill_TelModel!(om::OrderModel, lm::LinearModel, inds) =
	copy_to_LinearModel!(om.tel.lm, lm, inds)


"""
	fill_StarModel!(om, lm; inds=2:size(lm.M, 2))

Replace the stellar model in `om` with `lm`
"""
function fill_StarModel!(om::OrderModel, lm::FullLinearModel; inds=2:size(lm.M, 2))
	if length(inds) > 0; @assert inds[1] > 1 end
	copy_to_LinearModel!(om.star.lm, lm, inds)
	if typeof(om) <: OrderModelDPCA
		om.rv.lm.M .= view(lm.M, :, 1)
		om.rv.lm.s[:] .= view(lm.s, 1, :)
	else
		om.rv .= view(lm.s, 1, :) .* -light_speed_nu #TODO check if this is correct
	end
end
function fill_StarModel!(om::OrderModel, lm::LinearModel, rvs::Vector, inds)
	copy_to_LinearModel!(om.star.lm, lm, inds)
	if typeof(om) <: OrderModelDPCA
		om.rv.lm.M .= doppler_component(om.star.λ, om.star.lm.μ)
		om.rv.lm.s[:] .= rvs ./ -light_speed_nu #TODO check if this is correct
	else
		om.rv .= rvs
	end
end


"""
	fill_OrderModel!(om1, om2, inds_tel, inds_star)

Replace the models in `om1` with those in `om2`
"""
function fill_OrderModel!(om1::OrderModel, om2::OrderModel, inds_tel, inds_star)
	fill_TelModel!(om1, om2.tel.lm, inds_tel)
	if typeof(om2) <: OrderModelWobble
		fill_StarModel!(om1, om2.star.lm, om2.rv, inds_star)
	else
		fill_StarModel!(om1, om2.star.lm, om2.rv.lm.s, inds_star)
	end
end


"L1 norm"
L1(a) = sum(abs, a)
"L2 norm"
L2(a) = sum(abs2, a)
"L∞ norm"
L∞(Δ::VecOrMat{<:Real}) = maximum(Δ)
L∞(Δ) = maximum(L∞.(Δ))


"""
	shared_attention(M)

Ad-hoc regularization that punishes feature vectors with power in the same place
"""
function shared_attention(M)
	shared_attentions = M' * M
	return sum(shared_attentions) - sum(diag(shared_attentions))
end


"""
	model_prior(lm, om, key)

Calculate the model prior on `lm` with the regularization terms in `om.reg_` * `key`.
Can optionally exclude a number of the first basis vectors.
"""
function model_prior(lm, om::OrderModel, key::Symbol; exclude_vectors::Int64 = 0)

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

	if isFullLinearModel && (size(lm[1], 2) > exclude_vectors)

		lm_M = (exclude_vectors > 0 ? lm[1][:, (1+exclude_vectors):size(lm[1], 2)] : lm[1])
		lm_s = (exclude_vectors > 0 ? lm[2][(1+exclude_vectors):size(lm[2], 1), :] : lm[2])

		if haskey(reg, :shared_M); val += shared_attention(lm_M) * reg[:shared_M] end
		if haskey(reg, :L2_M); val += L2(lm_M) * reg[:L2_M] end
		if haskey(reg, :L1_M); val += L1(lm_M) * reg[:L1_M] end
		# if haskey(reg, :GP_μ); val -= gp_ℓ_precalc(sm.Δℓ_coeff, view(lm[1], :, 1), sm.A_sde, sm.Σ_sde) * reg[:GP_μ] end
		if haskey(reg, :GP_M)
			for i in 1:size(lm_M, 2)
				val -= gp_ℓ_precalc(sm.Δℓ_coeff, lm_M[:, i], sm.A_sde, sm.Σ_sde) * reg[:GP_M]
			end
		end
		val += model_s_prior(lm_s, reg)

	end

	return val
end
model_prior(lm::Union{FullLinearModel, TemplateModel}, om::OrderModel, key::Symbol) = model_prior(vec(lm), om, key)

nonzero_key(reg, key) = haskey(reg, key) && reg[key] != 0

"""
	model_s_prior(s, reg)

Add an L2 term to the scores if there are any regularization terms applied to the feature vectors
"""
function model_s_prior(s, reg::Dict)
	if (nonzero_key(reg, :L1_M) || nonzero_key(reg, :L2_M) || nonzero_key(reg, :GP_M))
		return L2(s)
	end
	return 0
end

"""
	tel_prior(om)

Calculate the telluric model prior on `om.tel.lm` with the regularization terms in `om.reg_model`
"""
function tel_prior(lm, om::OrderModel)

	if om.metadata[:tel_prior]

		exclude_vectors = (om.metadata[:h2o_vector] > 0) + (om.metadata[:oxy_vector] > 0)
		val = model_prior(lm, om, :tel; exclude_vectors=exclude_vectors)
	
		if haskey(om.reg_model, :L1_μ)
			val += L1(om.metadata[:tel_prior_spec] - lm[min(3, length(lm))]) * om.reg_model[:L1_μ]
		end

		if haskey(om.reg_model, :L2_μ)
			val += L2(om.metadata[:tel_prior_spec] - lm[min(3, length(lm))]) * om.reg_model[:L2_μ]
		end

		if haskey(om.reg_model, :GP_μ)
			μ_mod = lm[min(3, length(lm))] - om.metadata[:tel_prior_spec]
			val -= gp_ℓ_precalc(om.tel.Δℓ_coeff, μ_mod, om.tel.A_sde, om.tel.Σ_sde) * om.reg_model[:GP_μ]
		end

		if length(lm) >= 3
			if om.metadata[:h2o_vector] >= 1 && size(lm[1], 2) >= om.metadata[:h2o_vector]
				if haskey(om.reg_model, :L1_h2o)
					val += L1(log.(om.metadata[:h2o_prior_spec]) - lm[1][:,om.metadata[:h2o_vector]]) * om.reg_model[:L1_h2o]
				end
				if haskey(om.reg_model, :L2_h2o)
					val += L2(log.(om.metadata[:h2o_prior_spec]) - lm[1][:,om.metadata[:h2o_vector]]) * om.reg_model[:L2_h2o]
				end
				if haskey(om.reg_model, :GP_h2o)
					M_mod = lm[1][:,om.metadata[:h2o_vector]] - log.(om.metadata[:h2o_prior_spec])
					val -= gp_ℓ_precalc(om.tel.Δℓ_coeff, M_mod, om.tel.A_sde, om.tel.Σ_sde) * om.reg_model[:GP_h2o]
				end
			end
			if om.metadata[:oxy_vector] >= 1 && size(lm[1], 2) >= om.metadata[:oxy_vector] 
				if haskey(om.reg_model, :L1_oxy)
					val += L1(log.(om.metadata[:oxy_prior_spec]) - lm[1][:,om.metadata[:oxy_vector]]) * om.reg_model[:L1_oxy]
				end
				if haskey(om.reg_model, :L2_oxy)
					val += L2(log.(om.metadata[:oxy_prior_spec]) - lm[1][:,om.metadata[:oxy_vector]]) * om.reg_model[:L2_oxy]
				end
				if haskey(om.reg_model, :GP_oxy)
					M_mod = lm[1][:,om.metadata[:oxy_vector]] - log.(om.metadata[:oxy_prior_spec])
					val -= gp_ℓ_precalc(om.tel.Δℓ_coeff, M_mod, om.tel.A_sde, om.tel.Σ_sde) * om.reg_model[:GP_oxy]
				end
			end
		end
	
	else

		val = model_prior(lm, om, :tel)

	end

	if om.metadata[:tel_below_one]
		val += sum(exp.((lm[min(3, length(lm))] .- 1) .* (lm[min(3, length(lm))] .> 1)) .- 1) * 1e10
	end

	return val

end
tel_prior(lm::Union{FullLinearModel, TemplateModel}, om::OrderModel) = tel_prior(vec(lm), om)
tel_prior(om::OrderModel) = tel_prior(om.tel.lm, om)

"""
	star_prior(om)

Calculate the stellar model prior on `om.star.lm` with the regularization terms in `om.star_tel`
"""
function star_prior(lm, om::OrderModel)

	val = model_prior(lm, om, :star)

	if om.metadata[:tel_prior] && haskey(om.reg_model, :L2_μ_star) && haskey(om.metadata[:fixed_vectors], "log_λ") && haskey(om.metadata[:fixed_vectors], "star_μ")
		_flux_star_interp = CubicSpline(om.metadata[:fixed_vectors]["star_μ"], om.metadata[:fixed_vectors]["log_λ"]; extrapolate=true)
		flux_star_interp = _flux_star_interp(om.star.log_λ)
		flux_star_interp[om.star.log_λ .< minimum(om.metadata[:fixed_vectors]["log_λ"])] .= 1
		flux_star_interp[om.star.log_λ .> maximum(om.metadata[:fixed_vectors]["log_λ"])] .= 1
		val += L2(flux_star_interp - lm[min(3, length(lm))]) * om.reg_model[:L2_μ_star]
	end

	if om.metadata[:tel_prior] && haskey(om.reg_model, :GP_μ_star)
		μ_mod = lm[min(3, length(lm))] .- 1
		val -= gp_ℓ_precalc(om.star.Δℓ_coeff, μ_mod, om.star.A_sde, om.star.Σ_sde) * om.reg_model[:GP_μ_star]
	end

	return val

end
star_prior(om::OrderModel) = star_prior(om.star.lm, om)
#star_prior(lm, om::OrderModel) = model_prior(lm, om, :star)


"""
	total_model(tel, star)

Multiply the telluric and stellar models
"""
total_model(tel, star, rv) = tel .* (star .+ rv)
total_model(tel, star) = tel .* star


"""
	OutputDPCA

Holds the current outputs for the models at the observed wavelengths
"""
struct OutputDPCA{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}} <: Output
	"Telluric model at the observed wavelengths"
	tel::AM
	"Stellar model at the observed wavelengths"
	star::AM
	"RV model at the observed wavelengths"
	rv::AM
	"Total model at the observed wavelengths"
	total::M
	function OutputDPCA(tel::AM, star::AM, rv::AM, total::M) where {T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}}
		@assert size(tel) == size(star) == size(rv) == size(total)
		new{T, AM, M}(tel, star, rv, total)
	end
end


"""
	Output(om, d)

Calculates the current outputs for the models at the observed wavelengths
"""
function Output(om::OrderModelDPCA, d::Data)
	@assert size(om.b2o[1], 1) == size(d.flux, 1)
	return OutputDPCA(tel_model(om), star_model(om), rv_model(om), d)
end
OutputDPCA(tel, star, rv, d::GenericData) =
	OutputDPCA(tel, star, rv, total_model(tel, star, rv))
OutputDPCA(tel, star, rv, d::LSFData) =
	OutputDPCA(tel, star, rv, spectra_interp(total_model(tel, star, rv), d.lsf))

Base.copy(o::OutputDPCA) = OutputDPCA(copy(tel), copy(star), copy(rv))


"""
	recalc_total!(o, d)

Recalulates the current outputs for the total model at the observed wavelengths
"""
function recalc_total!(o::OutputDPCA, d::GenericData)
	o.total .= total_model(o.tel, o.star, o.rv)
end
function recalc_total!(o::OutputDPCA, d::LSFData)
	o.total .= spectra_interp(total_model(o.tel, o.star, o.rv), d.lsf)
end


"""
	Output!(o, om, d)

Recalculates the current outputs for the models at the observed wavelengths
"""
function Output!(o::OutputDPCA, om::OrderModelDPCA, d::Data)
	o.tel .= tel_model(om)
	o.star .= star_model(om)
	o.rv .= rv_model(om)
	recalc_total!(o, d)
end


"""
	OutputWobble

Holds the current outputs for the models at the observed wavelengths
"""
struct OutputWobble{T<:Number, AM<:AbstractMatrix{T}, M<:Matrix{T}} <: Output
	"Telluric model at the observed wavelengths"
	tel::AM
	"Stellar model at the observed wavelengths"
	star::AM
	"Total model at the observed wavelengths"
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
	OutputWobble(tel, star, spectra_interp(total_model(tel, star), d.lsf))


Base.copy(o::OutputWobble) = OutputWobble(copy(tel), copy(star))


function recalc_total!(o::OutputWobble, d::GenericData)
	o.total .= total_model(o.tel, o.star)
end
function recalc_total!(o::OutputWobble, d::LSFData)
	o.total .= spectra_interp(total_model(o.tel, o.star), d.lsf)
end


function Output!(o::OutputWobble, om::OrderModelWobble, d::Data)
	o.tel .= tel_model(om)
	o.star .= star_model(om)
	recalc_total!(o, d)
end


"""
	copy_reg!(from, to)

Copy the regularizations from one OrderModel to another
"""
function copy_reg!(from::OrderModel, to::OrderModel)
	copy_dict!(to.reg_tel, from.reg_tel)
	copy_dict!(to.reg_star, from.reg_star)
end


"""
	no_tellurics(model)

Figure out whether a `model` uses its telluric model
"""
no_tellurics(model::OrderModel) = all(isone.(model.tel.lm.μ)) && !is_time_variable(model.tel)