using AbstractGPs
using KernelFunctions
using TemporalGPs
using Distributions
import Base.copy
using BandedMatrices
using SpecialFunctions

abstract type Data end

struct LSFData{T<:Real} <: Data
    flux::AbstractMatrix{T}
    var::AbstractMatrix{T}
    log_λ_obs::AbstractMatrix{T}
	log_λ_obs_bounds::AbstractMatrix{T}
    log_λ_star::AbstractMatrix{T}
	log_λ_star_bounds::AbstractMatrix{T}
	lsf_broadener::AbstractVector{BandedMatrix}
	function LSFData(flux::AbstractMatrix{T}, var, log_λ_obs, log_λ_star, lsf_broadener) where {T<:Real}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		@assert length(lsf_broadener) == size(flux, 2)
		return new{T}(flux, var, log_λ_obs, bounds_generator(log_λ_obs), log_λ_star, bounds_generator(log_λ_star), lsf_broadener)
	end
end
(d::LSFData)(inds::AbstractVecOrMat) =
	LSFData(view(d.flux, :, inds), view(d.var, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds), view(lsf_broadener, inds))
Base.copy(d::LSFData) = LSFData(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star), copy(lsf_broadener))

struct GenericData{T<:Real} <: Data
    flux::AbstractMatrix{T}
    var::AbstractMatrix{T}
    log_λ_obs::AbstractMatrix{T}
	log_λ_obs_bounds::AbstractMatrix{T}
    log_λ_star::AbstractMatrix{T}
	log_λ_star_bounds::AbstractMatrix{T}
	function GenericData(flux::AbstractMatrix{T}, var, log_λ_obs, log_λ_star) where {T<:Real}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		return new{T}(flux, var, log_λ_obs, bounds_generator(log_λ_obs), log_λ_star, bounds_generator(log_λ_star))
	end
end
(d::GenericData)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, :, inds), view(d.var, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds))
Base.copy(d::GenericData) = GenericData(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star))

function create_λ_template(log_λ_obs::AbstractMatrix; upscale::Real=2*sqrt(2))
    log_min_wav, log_max_wav = [minimum(log_λ_obs), maximum(log_λ_obs)]
    Δ_logλ_og = minimum(log_λ_obs[end, :] - log_λ_obs[1, :]) / size(log_λ_obs, 1)
	Δ_logλ = Δ_logλ_og / upscale
    log_λ_template = (log_min_wav - 2 * Δ_logλ_og):Δ_logλ:(log_max_wav + 2 * Δ_logλ_og)
    λ_template = exp.(log_λ_template)
    return log_λ_template, λ_template
end

abstract type LinearModel end

# Full (includes mean) linear model
struct FullLinearModel{T<:Number} <: LinearModel
	M::AbstractMatrix{T}
	s::AbstractMatrix{T}
	μ::Vector{T}
	function FullLinearModel(M::AbstractMatrix{T}, s, μ) where {T<:Number}
		@assert length(μ) == size(M, 1)
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s, μ)
	end
end
Base.copy(flm::FullLinearModel) = FullLinearModel(copy(flm.M), copy(flm.s), copy(flm.μ))
LinearModel(flm::FullLinearModel, inds::AbstractVecOrMat) =
	FullLinearModel(flm.M, view(flm.s, :, inds), flm.μ)

# Base (no mean) linear model
struct BaseLinearModel{T<:Number} <: LinearModel
	M::AbstractMatrix{T}
	s::AbstractMatrix{T}
	function BaseLinearModel(M::AbstractMatrix{T}, s) where {T<:Number}
		@assert size(M, 2) == size(s, 1)
		return new{T}(M, s)
	end
end
Base.copy(blm::BaseLinearModel) = BaseLinearModel(copy(blm.M), copy(blm.s))
LinearModel(blm::BaseLinearModel, inds::AbstractVecOrMat) =
	BaseLinearModel(blm.M, view(blm.s, :, inds))

# Template (no bases) model
struct TemplateModel{T<:Number} <: LinearModel
	μ::Vector{T}
	n::Int
	TemplateModel(μ::Vector{T}, n) where {T<:Number} = new{T}(μ, n)
end
Base.copy(tlm::TemplateModel) = TemplateModel(copy(tlm.μ), tlm.n)
LinearModel(tm::TemplateModel, inds::AbstractVecOrMat) = tm

LinearModel(M::AbstractMatrix, s::AbstractMatrix, μ::AbstractVector) = FullLinearModel(M, s, μ)
LinearModel(M::AbstractMatrix, s::AbstractMatrix) = BaseLinearModel(M, s)
LinearModel(μ::AbstractVector, n::Int) = TemplateModel(μ, n)

LinearModel(lm::FullLinearModel, s::AbstractMatrix) = FullLinearModel(lm.M, s, lm.μ)
LinearModel(lm::BaseLinearModel, s::AbstractMatrix) = BaseLinearModel(lm.M, s)
LinearModel(lm::TemplateModel, s::AbstractMatrix) = lm

_eval_blm(M::AbstractVecOrMat, s::AbstractVecOrMat) = M * s
_eval_flm(M::AbstractVecOrMat, s::AbstractVecOrMat, μ::AbstractVector) =
	_eval_blm(M, s) .+ μ
_eval_lm(M::AbstractVecOrMat, s::AbstractVecOrMat) = _eval_blm(M, s)
_eval_lm(M::AbstractVecOrMat, s::AbstractVecOrMat, μ::AbstractVector) = _eval_flm(M, s, μ)

(flm::FullLinearModel)() = _eval_flm(flm.M, flm.s, flm.μ)
(blm::BaseLinearModel)() = _eval_blm(blm.M, blm.s)
(tlm::TemplateModel)() = repeat(tlm.μ, 1, tlm.n)
(flm::FullLinearModel)(inds::AbstractVecOrMat) = _eval_flm(view(flm.M, inds, :), flm.s, flm.μ)
(blm::BaseLinearModel)(inds::AbstractVecOrMat) = _eval_blm(view(blm.M, inds, :), blm.s)
(tlm::TemplateModel)(inds::AbstractVecOrMat) = repeat(view(tlm.μ, inds), 1, tlm.n)

function copy_LinearModel!(from::LinearModel, to::LinearModel)
	@assert typeof(to)==typeof(from)
	for i in fieldnames(typeof(from))
		getfield(to, i)[:] = getfield(from, i)
	end
end

struct Submodel{T<:Number}
    log_λ::AbstractVector{T}
    λ::AbstractVector{T}
	lm::LinearModel
    function Submodel(log_λ_obs::AbstractVecOrMat, n_comp::Int; include_mean::Bool=true, kwargs...)
        n_obs = size(log_λ_obs, 2)
		log_λ, λ = create_λ_template(log_λ_obs; kwargs...)
		len = length(log_λ)
		if include_mean
			lm = FullLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs), ones(len))
		else
			lm = BaseLinearModel(zeros(len, n_comp), zeros(n_comp, n_obs))
		end
        return Submodel(log_λ, λ, lm)
    end
    function Submodel(log_λ::AbstractVector{T}, λ, lm) where {T<:Number}
		if typeof(lm) <: TemplateModel
			@assert length(log_λ) == length(λ) == length(lm.μ)
        else
			@assert length(log_λ) == length(λ) == size(lm.M, 1)
		end
        return new{T}(log_λ, λ, lm)
    end
end
(sm::Submodel)(inds::AbstractVecOrMat) =
	Submodel(sm.log_λ, sm.λ, LinearModel(sm.lm, inds))
(sm::Submodel)() = sm.lm()
Base.copy(sm::Submodel) = Submodel(sm.log_λ, sm.λ, copy(sm.lm))

function _shift_log_λ_model(log_λ_obs_from, log_λ_obs_to, log_λ_model_from)
	n_obs = size(log_λ_obs_from, 2)
	dop = [log_λ_obs_from[1, i] - log_λ_obs_to[1, i] for i in 1:n_obs]
	log_λ_model_to = ones(length(log_λ_model_from), n_obs)
	for i in 1:n_obs
		log_λ_model_to[:, i] = log_λ_model_from .+ dop[i]
	end
	return log_λ_model_to
end

default_reg_tel = Dict([(:L2_μ, 1e6), (:L1_μ, 1e2),
	(:L1_μ₊_factor, 6), (:L2_M, 1e-1), (:L1_M, 1e3)])
default_reg_star = Dict([(:L2_μ, 1e4), (:L1_μ, 1e3),
	(:L1_μ₊_factor, 7.2), (:L2_M, 1e1), (:L1_M, 1e6)])

struct OrderModel{T<:Number}
    tel::Submodel{T}
    star::Submodel{T}
	rv::Submodel{T}
	reg_tel::Dict{Symbol, T}
	reg_star::Dict{Symbol, T}
	metadata::Dict{Symbol, Any}
    function OrderModel(
		d::Data,
		instrument::String,
		order::Int,
		star::String;
		n_comp_tel::Int=2,
		n_comp_star::Int=2,
		kwargs...)

        tel = Submodel(d.log_λ_obs, n_comp_tel; kwargs...)
        star = Submodel(d.log_λ_star, n_comp_star; kwargs...)
		rv = Submodel(d.log_λ_star, 1; include_mean=false, kwargs...)

        n_obs = size(d.log_λ_obs, 2)
        star_dop = [d.log_λ_star[1, i] - d.log_λ_obs[1, i] for i in 1:n_obs]
        star_log_λ_tel = ones(length(star.log_λ), n_obs)
        for i in 1:n_obs
            star_log_λ_tel[:, i] = star.log_λ .+ star_dop[i]
        end
		todo = Dict([(:reg_improved, false), (:downsized, false), (:optimized, false), (:err_estimated, false)])
		metadata = Dict([(:todo, todo), (:instrument, instrument), (:order, order), (:star, star)])
        return OrderModel(tel, star, rv, copy(default_reg_tel), copy(default_reg_star), metadata)
    end
    function OrderModel(tel::Submodel{T}, star, rv, reg_tel, reg_star, metadata) where {T<:Number}
		return new{T}(tel, star, rv, reg_tel, reg_star, metadata)
	end
end
Base.copy(om::OrderModel) = OrderModel(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), copy(om.metadata))
(om::OrderModel)(inds::AbstractVecOrMat) =
	OrderModel(om.tel(inds), om.star(inds), om.rv(inds), om.reg_tel,
	om.reg_star, copy(om.metadata))
function zero_regularization(om::OrderModel)
	for (key, value) in om.reg_tel
		om.reg_tel[key] = 0
	end
	for (key, value) in om.reg_star
		om.reg_star[key] = 0
	end
end

# I have no idea why the negative sign needs to be here
rvs(model::OrderModel) = vec(Array((model.rv.lm.s .* -light_speed_nu)'))

function downsize(lm::FullLinearModel, n_comp::Int)
	if n_comp > 0
		return FullLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :], lm.μ[:])
	else
		return TemplateModel(lm.μ[:], size(lm.M, 1))
	end
end
downsize(lm::BaseLinearModel, n_comp::Int) =
	BaseLinearModel(lm.M[:, 1:n_comp], lm.s[1:n_comp, :])
downsize(sm::Submodel, n_comp::Int) =
	Submodel(sm.log_λ[:], sm.λ[:], downsize(sm.lm, n_comp))
downsize(m::OrderModel, n_comp_tel::Int, n_comp_star::Int) =
	OrderModel(
		downsize(m.tel, n_comp_tel),
		downsize(m.star, n_comp_star),
		m.rv, m.reg_tel, m.reg_star, m.metadata)

tel_prior(om::OrderModel) = model_prior(om.tel.lm, om.reg_tel)
star_prior(om::OrderModel) = model_prior(om.star.lm, om.reg_star)

spectra_interp(vals::AbstractVector, basis::AbstractVector, bounds::AbstractVector) =
	[oversamp_interp(bounds[i], bounds[i+1], basis, vals) for i in 1:(length(bounds)-1)]
function spectra_interp(vals::AbstractMatrix, basis::AbstractVector, bounds::AbstractMatrix)
	interped_vals = zeros(size(bounds, 1)-1, size(bounds, 2))
	for i in 1:size(interped_vals, 2)
		interped_vals[:, i] = spectra_interp(view(vals, :, i), basis, view(bounds, :, i))
	end
	return interped_vals
end

tel_model(om::OrderModel, d::Data) = spectra_interp(om.tel.lm(), om.tel.log_λ, d.log_λ_obs_bounds)
star_model(om::OrderModel, d::Data) = spectra_interp(om.star.lm(), om.star.log_λ, d.log_λ_star_bounds)
rv_model(om::OrderModel, d::Data) = spectra_interp(om.rv.lm(), om.rv.log_λ, d.log_λ_star_bounds)


function fix_FullLinearModel_s!(flm, min::Number, max::Number)
	@assert all(min .< flm.μ .< max)
	result = ones(typeof(flm.μ[1]), length(flm.μ))
	for i in 1:size(flm.s, 2)
		result[:] = _eval_flm(flm.M, flm.s[:, i], flm.μ)
		while any(result .> max) || any(result .< min)
			# println("$i, old s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
			flm.s[:, i] ./= 2
			result[:] = _eval_flm(flm.M, flm.s[:, i], flm.μ)
			# println("$i, new s: $(lm.s[:, i]), min: $(minimum(result)), max:  $(maximum(result))")
		end
	end
end

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
	f_naive = GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
	return to_sde(f_naive, SArrayStorage(Float64))
end

SOAP_gp_params = (var_kernel = 3.3270754364467443, λ = 1 / 9.021560480866474e-5)
SOAP_gp = build_gp(SOAP_gp_params)

# ParameterHandling version
# SOAP_gp_params = (var_kernel = positive(3.3270754364467443), λ = positive(1 / 9.021560480866474e-5))
# flat_SOAP_gp_params, unflatten = value_flatten(SOAP_gp_params)
# # unflatten(flat_SOAP_gp_params) == ParameterHandling.value(SOAP_gp_params)  # true
# SOAP_gp = build_gp(ParameterHandling.value(SOAP_gp_params))

function _spectra_interp_gp!(fluxes, vars, log_λ, flux_obs, var_obs, log_λ_obs; gp_mean::Number=1.)
	for i in 1:size(flux_obs, 2)
		gp = get_marginal_GP(SOAP_gp(log_λ_obs[:, i], var_obs[:, i]), flux_obs[:, i] .- gp_mean, log_λ)
		fluxes[:, i] = mean.(gp) .+ gp_mean
        vars[:, i] = var.(gp)
	end
end

function _spectra_interp_gp_div_gp!(fluxes::AbstractMatrix, vars::AbstractMatrix, log_λ::AbstractVector, flux_obs::AbstractMatrix, var_obs::AbstractMatrix, log_λ_obs::AbstractMatrix, flux_other::AbstractMatrix, var_other::AbstractMatrix, log_λ_other::AbstractMatrix; gp_mean::Number=1.)
	for i in 1:size(flux_obs, 2)
		gpn = get_marginal_GP(SOAP_gp(log_λ_obs[:, i], var_obs[:, i]), flux_obs[:, i] .- gp_mean, log_λ)
		gpd = get_marginal_GP(SOAP_gp(log_λ_other[:, i], var_other[:, i]), flux_other[:, i] .- gp_mean, log_λ)
		gpn_μ = mean.(gpn) .+ gp_mean
		gpd_μ = mean.(gpd) .+ gp_mean
		fluxes[:, i] = gpn_μ ./ gpd_μ
        vars[:, i] = (var.(gpn) .+ ((gpn_μ .^ 2 .* var.(gpd)) ./ (gpd_μ .^2))) ./ (gpd_μ .^2)
	end
end

# function n_comps_needed(sm::Submodel; threshold::Real=0.05)
#     @assert 0 < threshold < 1
#     s_var = sum(abs2, sm.lm.s; dims=2)
#     return findfirst(s_var ./ sum(s_var) .< threshold)[1] - 1
# end

function initialize!(om::OrderModel, d::Data; min::Number=0, max::Number=1.2, kwargs...)

	μ_min = min + 0.05
	μ_max = max - 0.05

	n_obs = size(d.flux, 2)
	n_comp_star = size(om.star.lm.M, 2) + 1
	n_comp_tel = size(om.tel.lm.M, 2)

	star_log_λ_tel = _shift_log_λ_model(d.log_λ_obs, d.log_λ_star, om.star.log_λ)
	tel_log_λ_star = _shift_log_λ_model(d.log_λ_star, d.log_λ_obs, om.tel.log_λ)
	flux_star = ones(length(om.star.log_λ), n_obs)
	vars_star = ones(length(om.star.log_λ), n_obs)
	flux_tel = ones(length(om.tel.log_λ), n_obs)
	vars_tel = ones(length(om.tel.log_λ), n_obs)
	_spectra_interp_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star)

	om.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	_, _, _, rvs_naive =
	    DPCA(flux_star, om.star.λ; template=om.star.lm.μ, num_components=1)

	# telluric model with stellar template
	flux_star .= om.star.lm.μ
	_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)

	om.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	# _, om.tel.lm.M[:, :], om.tel.lm.s[:, :], _ =
	#     fit_gen_pca(flux_tel; num_components=n_comp_tel, mu=om.tel.lm.μ)

	# stellar model with telluric template
	flux_tel .= om.tel.lm.μ
	_spectra_interp_gp_div_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)

	om.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	# _, M_star, s_star, rvs_notel =
	#     DEMPCA(flux_star, om.star.λ, 1 ./ vars_star; template=om.star.lm.μ, num_components=n_comp_star, kwargs...)
	_, M_star, s_star, rvs_notel =
		DEMPCA(flux_star, om.star.λ, 1 ./ vars_star; template=om.star.lm.μ, num_components=n_comp_star)
	fracvar_star = fracvar(flux_star .- om.star.lm.μ, M_star, s_star, 1 ./ vars_star)

	# telluric model with updated stellar template
	flux_star .= om.star.lm.μ
	_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)

	om.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	Xtmp = flux_tel .- om.tel.lm.μ
	# EMPCA!(om.tel.lm.M, Xtmp, om.tel.lm.s, 1 ./ vars_tel; kwargs...)
	EMPCA!(om.tel.lm.M, Xtmp, om.tel.lm.s, 1 ./ vars_tel)
	fracvar_tel = fracvar(Xtmp, om.tel.lm.M, om.tel.lm.s, 1 ./ vars_tel)

	om.star.lm.M[:, :], om.star.lm.s[:] = M_star[:, 2:end], s_star[2:end, :]
	om.rv.lm.M[:, :], om.rv.lm.s[:] = M_star[:, 1], s_star[1, :]'

	fix_FullLinearModel_s!(om.star.lm, min, max)
	fix_FullLinearModel_s!(om.tel.lm, min, max)

	return rvs_notel, rvs_naive, fracvar_tel, fracvar_star
end

L1(a::AbstractArray) = sum(abs.(a))
L2(a::AbstractArray) = sum(a .* a)
shared_attention(v1::AbstractVector, v2::AbstractVector) = dot(abs.(v1), abs.(v2))

function model_prior(lm::LinearModel, reg::Dict{Symbol, <:Real})
	val = 0
	if haskey(reg, :shared_M)
		shared_att = 0
		for i in size(lm.M, 2)
			for j in size(lm.M, 2)
				if i != j
					shared_att += shared_attention(lm.M[:, i], lm.M[:, j])
				end
			end
		end
		val += shared_att * reg[:shared_M]
	end
	if haskey(reg, :L2_μ) || haskey(reg, :L1_μ) || haskey(reg, :L1_μ₊_factor)
		μ_mod = lm.μ .- 1
		if haskey(reg, :L2_μ); val += L2(μ_mod) * reg[:L2_μ] end
		if haskey(reg, :L1_μ)
			val += L1(μ_mod) * reg[:L1_μ]
			if haskey(reg, :L1_μ₊_factor)
				val += sum(view(μ_mod, μ_mod .> 0)) * reg[:L1_μ₊_factor] * reg[:L1_μ]
			end
		end
	end
	if !(typeof(lm) <: TemplateModel)
		if haskey(reg, :L2_M); val += L2(lm.M) * reg[:L2_M] end
		if haskey(reg, :L1_M); val += L1(lm.M) * reg[:L1_M] end
		if (haskey(reg, :L1_M) && reg[:L1_M] != 0) || (haskey(reg, :L2_M) && reg[:L2_M] != 0); val += L1(lm.s) end
	end
	return val
end

struct Output{T<:Real}
	tel::AbstractMatrix{T}
	star::AbstractMatrix{T}
	rv::AbstractMatrix{T}
	total::AbstractMatrix{T}
	Output(om::OrderModel, d::Data) =
		Output(tel_model(om, d), star_model(om, d), rv_model(om, d), d)
	Output(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::GenericData) =
		Output(tel, star, rv, tel .* (star + rv))
	function Output(tel::AbstractMatrix{T}, star::AbstractMatrix{T}, rv::AbstractMatrix{T}, d::LSFData) where {T<:Real}
		total = tel .* (star + rv)
		for i in 1:size(total, 2)
			total[:, i] = d.lsf_broadener[i] * total[:, i]
		end
		return Output(tel, star, rv, total)
	end
	function Output(tel::AbstractMatrix{T}, star::AbstractMatrix{T}, rv::AbstractMatrix{T}, total::AbstractMatrix{T}) where {T<:Real}
		@assert size(tel) == size(star) == size(rv) == size(total)
		new{T}(tel, star, rv, total)
	end
end
Base.copy(o::Output) = Output(copy(tel), copy(star), copy(rv))
function recalc_total!(o::Output, d::GenericData)
	o.total[:] = o.tel .* (o.star + o.rv)
end
function recalc_total!(o::Output, d::LSFData)
	o.total[:] = o.tel .* (o.star + o.rv)
	for i in 1:size(model, 2)
		o.total[i, :] = d.lsf_broadener[i] * o.total[i, :]
	end
end

function copy_reg!(from::OrderModel, to::OrderModel)
	copy_dict!(from.reg_tel, to.reg_tel)
	copy_dict!(from.reg_star, to.reg_star)
end
