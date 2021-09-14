using AbstractGPs
using KernelFunctions
using TemporalGPs
using ParameterHandling
using Stheno
using Distributions
import Base.copy
using BandedMatrices
using SpecialFunctions

abstract type Data end

_EXPRES_lsf_FWHM = 4.5  # roughly. See http://exoplanets.astro.yale.edu/science/activity.php
_EXPRES_lsf_σ = _EXPRES_lsf_FWHM / (2 * sqrt(2 * log(2)))

struct EXPRESData{T<:Real} <: Data
    flux::AbstractMatrix{T}
    var::AbstractMatrix{T}
    log_λ_obs::AbstractMatrix{T}
    log_λ_star::AbstractMatrix{T}
	Σ_lsf::AbstractMatrix{T}
	function EXPRESData(flux::AbstractMatrix{T}, var, log_λ_obs, log_λ_star; span::Int=4) where {T<:Real}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		lsf_pm_4 = [erf((i+0.5)/_EXPRES_lsf_σ) - erf((i-0.5)/_EXPRES_lsf_σ) for i in -span:span]
		x = zeros(20, 20)
		xl = size(x, 2)
		for i in 1:xl
		    low = max(i - span, 1); high = min(i + span, xl);
		    x[low:high, i] = view(lsf_pm_4, (low + span + 1 - i):(high + span + 1 - i))
		    x[low:high, i] ./= sum(x[low:high, i])
		end
		Σ_lsf = BandedMatrix(x, (span, span))
		return new{T}(flux, var, log_λ_obs, log_λ_star, Σ_lsf)
	end
end
struct GenericData{T<:Real} <: Data
    flux::AbstractMatrix{T}
    var::AbstractMatrix{T}
    log_λ_obs::AbstractMatrix{T}
    log_λ_star::AbstractMatrix{T}
	function Data(flux::AbstractMatrix{T}, var, log_λ_obs, log_λ_star) where {T<:Real}
		@assert size(flux) == size(var) == size(log_λ_obs) == size(log_λ_star)
		return new{T}(flux, var, log_λ_obs, log_λ_star)
	end
end
(d::GenericData)(inds::AbstractVecOrMat) =
	GenericData(view(d.flux, :, inds), view(d.var, :, inds),
	view(d.log_λ_obs, :, inds), view(d.log_λ_star, :, inds))
Base.copy(d::GenericData) = GenericData(copy(d.flux), copy(d.var), copy(d.log_λ_obs), copy(d.log_λ_star))


function create_λ_template(log_λ_obs, resolution)
    log_min_wav, log_max_wav = [minimum(log_λ_obs), maximum(log_λ_obs)]
    len = Int(ceil((exp(log_max_wav) - exp(log_min_wav)) * resolution / exp((log_max_wav + log_min_wav)/2)))
    log_Δλ = (log_max_wav - log_min_wav) / len
    len += 2
    log_λ_template = range(log_min_wav - log_Δλ; length = len,  stop = log_max_wav + log_Δλ)
    λ_template = exp.(log_λ_template)
    return len, log_λ_template, λ_template
end

struct LinearInterpolationHelper{T<:Number}
    li::AbstractMatrix{Int}  # lower indices
    ratios::AbstractMatrix{T}
	function LinearInterpolationHelper(
		li::AbstractMatrix{Int},  # lower indices
	    ratios::AbstractMatrix{T}) where {T<:Number}
		@assert size(li) == size(ratios)
		@assert all(0 .<= ratios .<= 1)
		# @assert some issorted thing?
		return new{T}(li, ratios)
	end
end
function (lih::LinearInterpolationHelper)(inds::AbstractVecOrMat, n_in::Int)
	n_out, n_obs = size(lih.li)
	@assert all(0 .< inds .<= n_obs)
	@assert allunique(inds)
	difference = 0
	j = inds[1]-1
	new_li = ones(Int, n_out, length(inds))
	new_li[:, 1] = lih.li[:, inds[1]] .- (j * n_in)
	for i in 2:length(inds)
		j += (inds[i] - inds[i - 1]) - 1
		new_li[:, i] = lih.li[:, inds[i]] .- (j * n_in)
	end
	return LinearInterpolationHelper(new_li, view(lih.ratios, :, inds))
end


function LinearInterpolationHelper_maker(to_λs::AbstractVecOrMat, from_λs::AbstractMatrix)
	len_from, n_obs = size(from_λs)
	len_to = size(to_λs, 1)

	lower_inds_t2f = zeros(Int, (len_from, n_obs))
	lower_inds_f2t = zeros(Int, (len_to, n_obs))
	ratios_t2f = zeros((len_from, n_obs))
	ratios_f2t = zeros((len_to, n_obs))

	function helper!(j, len, lower_inds, ratios, λs1, λs2)
		lower_inds[:, j] = searchsortednearest(λs1, λs2; lower=true)
		for i in 1:size(lower_inds, 1)
			# if point is after the end, just say the point is at the end
			if lower_inds[i, j] >= len
				lower_inds[i, j] = len - 1
				ratios[i, j] = 1
			# if point is before the start, keep ratios to be 0
			elseif λs2[i] >= λs1[lower_inds[i, j]]
				x0 = λs1[lower_inds[i, j]]
				x1 = λs1[lower_inds[i, j]+1]
				ratios[i, j] = (λs2[i] - x0) / (x1 - x0)
			end
			@assert 0 <= ratios[i,j] <= 1 "something is off with ratios[$i,$j] = $(ratios[i,j])"
		end
	end

	for j in 1:n_obs
    	current_from_λs = view(from_λs, :, j)
		helper!(j, len_to, lower_inds_t2f, ratios_t2f, to_λs, current_from_λs)
		helper!(j, len_from, lower_inds_f2t, ratios_f2t, current_from_λs, to_λs)
		lower_inds_t2f[:, j] .+= (j - 1) * len_to
		lower_inds_f2t[:, j] .+= (j - 1) * len_from
	end

	return LinearInterpolationHelper(lower_inds_t2f, ratios_t2f), LinearInterpolationHelper(lower_inds_f2t, ratios_f2t)
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

# Template (no bases) model
struct TemplateModel{T<:Number} <: LinearModel
	μ::Vector{T}
	n::Int
	TemplateModel(μ::Vector{T}, n) where {T<:Number} = new{T}(μ, n)
end
Base.copy(tlm::TemplateModel) = TemplateModel(copy(tlm.μ), tlm.n)

LinearModel(blm::BaseLinearModel, inds::AbstractVecOrMat) =
	BaseLinearModel(blm.M, view(blm.s, :, inds))
_eval_blm(M::AbstractVecOrMat, s::AbstractVecOrMat) = M * s
_eval_flm(M::AbstractVecOrMat, s::AbstractVecOrMat, μ::AbstractVector) =
	_eval_blm(M, s) .+ μ

(flm::FullLinearModel)() = _eval_flm(flm.M, flm.s, flm.μ)
(blm::BaseLinearModel)() = _eval_blm(blm.M, blm.s)
(tlm::TemplateModel)() = repeat(tlm.μ, 1, tlm.n)
(flm::FullLinearModel)(inds::AbstractVecOrMat) = _eval_flm(view(flm.M, inds, :), flm.s, flm.μ)
(blm::BaseLinearModel)(inds::AbstractVecOrMat) = _eval_blm(view(blm.M, inds, :), blm.s)
(tlm::TemplateModel)(inds::AbstractVecOrMat) = repeat(view(tlm.μ, inds), 1, tlm.n)

struct Submodel{T<:Number}
    log_λ::AbstractVector{T}
    λ::AbstractVector{T}
	lm::LinearModel
    function Submodel(log_λ_obs::AbstractVecOrMat, model_res::Real, n_comp::Int; include_mean::Bool=true)
        n_obs = size(log_λ_obs, 2)
		len, log_λ, λ = create_λ_template(log_λ_obs, model_res)
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
    lih_t2b::LinearInterpolationHelper
    lih_b2t::LinearInterpolationHelper
    lih_o2b::LinearInterpolationHelper
    lih_b2o::LinearInterpolationHelper
    lih_t2o::LinearInterpolationHelper
    lih_o2t::LinearInterpolationHelper
	metadata::Dict{Symbol, Any}
    function OrderModel(
		d::Data,
		star_model_res::Number,
		tel_model_res::Number,
		instrument::String,
		order::Int,
		star::String;
		n_comp_tel::Int=2,
		n_comp_star::Int=2)

        tel = Submodel(d.log_λ_obs, tel_model_res, n_comp_tel)
        star = Submodel(d.log_λ_star, star_model_res, n_comp_star)
		rv = Submodel(d.log_λ_star, star_model_res, 1; include_mean=false)

        n_obs = size(d.log_λ_obs, 2)
        lih_t2o, lih_o2t = LinearInterpolationHelper_maker(tel.log_λ, d.log_λ_obs)
        lih_b2o, lih_o2b = LinearInterpolationHelper_maker(star.log_λ, d.log_λ_star)
        star_dop = [d.log_λ_star[1, i] - d.log_λ_obs[1, i] for i in 1:n_obs]
        star_log_λ_tel = ones(length(star.log_λ), n_obs)
        for i in 1:n_obs
            star_log_λ_tel[:, i] = star.log_λ .+ star_dop[i]
        end
        lih_t2b, lih_b2t = LinearInterpolationHelper_maker(tel.log_λ, star_log_λ_tel)
		todo = Dict([(:reg_improved, false), (:downsized, false), (:optimized, false), (:err_estimated, false)])
		metadata = Dict([(:todo, todo), (:instrument, instrument), (:order, order), (:star, star)])
        return OrderModel(tel, star, rv, copy(default_reg_tel), copy(default_reg_star), lih_t2b, lih_b2t,
			lih_o2b, lih_b2o, lih_t2o, lih_o2t, metadata)
    end
    function OrderModel(tel::Submodel{T}, star, rv, reg_tel, reg_star, lih_t2b,
		lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t, metadata) where {T<:Number}
		return new{T}(tel, star, rv, reg_tel, reg_star, lih_t2b, lih_b2t, lih_o2b, lih_b2o, lih_t2o, lih_o2t, metadata)
	end
end
Base.copy(om::OrderModel) = OrderModel(copy(om.tel), copy(om.star), copy(om.rv), copy(om.reg_tel), copy(om.reg_star), om.lih_t2b,
	om.lih_b2t, om.lih_o2b, om.lih_b2o, om.lih_t2o, om.lih_o2t, copy(om.metadata))
(om::OrderModel)(inds::AbstractVecOrMat) =
	OrderModel(om.tel(inds), om.star(inds), om.rv(inds), om.reg_tel, om.reg_star,
	om.lih_t2b(inds, size(om.lih_o2t.li, 1)), om.lih_b2t(inds, size(om.lih_o2b.li, 1)),
	om.lih_o2b(inds, size(om.lih_b2o.li, 1)), om.lih_b2o(inds, size(om.lih_o2b.li, 1)),
	om.lih_t2o(inds, size(om.lih_o2t.li, 1)), om.lih_o2t(inds, size(om.lih_b2o.li, 1)),
	copy(om.metadata))
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
		m.rv, m.reg_tel, m.reg_star, m.lih_t2b, m.lih_b2t,
		m.lih_o2b, m.lih_b2o, m.lih_t2o, m.lih_o2t, m.metadata)

tel_prior(om::OrderModel) = model_prior(om.tel.lm, om.reg_tel)
star_prior(om::OrderModel) = model_prior(om.star.lm, om.reg_star)

spectra_interp(og_vals::AbstractMatrix, lih) =
    (og_vals[lih.li] .* (1 .- lih.ratios)) + (og_vals[lih.li .+ 1] .* (lih.ratios))

tel_model(om) = spectra_interp(om.tel(), om.lih_t2o)
star_model(om) = spectra_interp(om.star(), om.lih_b2o)
rv_model(om) = spectra_interp(om.rv(), om.lih_b2o)


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

SOAP_gp_params, unflatten = flatten((
    var_kernel = positive(3.3270754364467443),
    λ = positive(1 / 9.021560480866474e-5),
))

function build_gp(params)
    f_naive = GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
    return to_sde(f_naive, SArrayStorage(Float64))
end

unpack = ParameterHandling.value ∘ unflatten
params = unpack(SOAP_gp_params)
SOAP_gp = build_gp(params)

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

function n_comps_needed(sm::Submodel; threshold::Real=0.05)
    @assert 0 < threshold < 1
    s_var = sum(abs2, sm.lm.s; dims=2)
    return findfirst(s_var ./ sum(s_var) .< threshold)[1] - 1
end

function initialize!(om::OrderModel, d::Data; min::Number=0, max::Number=1.2, use_gp::Bool=false, kwargs...)

	μ_min = min + 0.05
	μ_max = max - 0.05

	n_obs = size(d.flux, 2)
	n_comp_star = size(om.star.lm.M, 2) + 1
	n_comp_tel = size(om.tel.lm.M, 2)

	if use_gp
		star_log_λ_tel = _shift_log_λ_model(d.log_λ_obs, d.log_λ_star, om.star.log_λ)
		tel_log_λ_star = _shift_log_λ_model(d.log_λ_star, d.log_λ_obs, om.tel.log_λ)
		flux_star = ones(length(om.star.log_λ), n_obs)
		vars_star = ones(length(om.star.log_λ), n_obs)
		flux_tel = ones(length(om.tel.log_λ), n_obs)
		vars_tel = ones(length(om.tel.log_λ), n_obs)
		_spectra_interp_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star)
	else
		flux_star = spectra_interp(d.flux, om.lih_o2b)
	end
	om.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	_, _, _, rvs_naive =
	    DPCA(flux_star, om.star.λ; template=om.star.lm.μ, num_components=1)

	# telluric model with stellar template
	if use_gp
		flux_star .= om.star.lm.μ
		_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
	else
		flux_tel = spectra_interp(d.flux, om.lih_o2t) ./
			spectra_interp(repeat(om.star.lm.μ, 1, n_obs), om.lih_b2t)
	end
	om.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	# _, om.tel.lm.M[:, :], om.tel.lm.s[:, :], _ =
	#     fit_gen_pca(flux_tel; num_components=n_comp_tel, mu=om.tel.lm.μ)

	# stellar model with telluric template
	if use_gp
		flux_tel .= om.tel.lm.μ
		_spectra_interp_gp_div_gp!(flux_star, vars_star, om.star.log_λ, d.flux, d.var, d.log_λ_star, flux_tel, vars_tel, tel_log_λ_star)
	else
		_tel_μ = spectra_interp(repeat(om.tel.lm.μ, 1, n_obs), om.lih_t2b)
		flux_star = spectra_interp(d.flux, om.lih_o2b) ./ _tel_μ
		vars_star = spectra_interp(d.var, om.lih_o2b) ./ (_tel_μ .^2)  # TODO:should be adding in quadtrature, but fix later
	end
	om.star.lm.μ[:] = make_template(flux_star; min=μ_min, max=μ_max)
	_, M_star, s_star, rvs_notel =
	    DEMPCA(flux_star, om.star.λ, 1 ./ vars_star; template=om.star.lm.μ, num_components=n_comp_star, kwargs...)
	fracvar_star = fracvar(flux_star .- om.star.lm.μ, M_star, s_star, 1 ./ vars_star)

	# telluric model with updated stellar template
	if use_gp
		flux_star .= om.star.lm.μ
		_spectra_interp_gp_div_gp!(flux_tel, vars_tel, om.tel.log_λ, d.flux, d.var, d.log_λ_obs, flux_star, vars_star, star_log_λ_tel)
	else
		_star_μ = spectra_interp(repeat(om.star.lm.μ, 1, n_obs), om.lih_b2t)
		flux_tel = spectra_interp(d.flux, om.lih_o2t) ./ _star_μ
		vars_tel = spectra_interp(d.var, om.lih_o2t) ./ (_star_μ .^ 2) # TODO:should be adding in quadtrature, but fix later
	end
	om.tel.lm.μ[:] = make_template(flux_tel; min=μ_min, max=μ_max)
	Xtmp = flux_tel .- om.tel.lm.μ
	EMPCA!(om.tel.lm.M, Xtmp, om.tel.lm.s, 1 ./ vars_tel; kwargs...)
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
	Output(om::OrderModel) = Output(tel_model(om), star_model(om), rv_model(om))
	function Output(tel::AbstractMatrix{T}, star, rv) where {T<:Real}
		@assert size(tel) == size(star) == size(rv)
		new{T}(tel, star, rv)
	end
end
Base.copy(o::Output) = Output(copy(tel), copy(star), copy(rv))

function copy_reg!(from::OrderModel, to::OrderModel)
	copy_dict!(from.reg_tel, to.reg_tel)
	copy_dict!(from.reg_star, to.reg_star)
end
