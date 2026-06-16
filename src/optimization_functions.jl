# using LineSearches
using ParameterHandling
using Optim
import Base.println
using DataInterpolations
import ExpectationMaximizationPCA as EMPCA

abstract type ModelWorkspace end
abstract type AdamWorkspace<:ModelWorkspace end
abstract type OptimWorkspace<:ModelWorkspace end

# generic χ² loss functions
_χ²_loss_σ(model_m_data, sigma) = (model_m_data ./ sigma) .^ 2
_χ²_loss(model_m_data, variance) = ((model_m_data) .^ 2) ./ variance
_χ²_loss(model, data, variance) = _χ²_loss(model .- data, variance)
_χ²_loss(model, data::Data; use_var_s::Bool=false) = use_var_s ? _χ²_loss(model, data.flux, data.var_s) : _χ²_loss(model, data.flux, data.var)

# χ² loss functions per pixel
__loss_diagnostic(tel, star, rv, d::GenericData; kwargs...) =
	_χ²_loss(total_model(tel, star, rv), d; kwargs...)
__loss_diagnostic(tel, star, rv, d::LSFData; kwargs...) =
	_χ²_loss(spectra_interp(total_model(tel, star, rv), d.lsf), d; kwargs...)
__loss_diagnostic(tel, star, d::GenericData; kwargs...) =
	_χ²_loss(total_model(tel, star), d; kwargs...)
__loss_diagnostic(tel, star, d::LSFData; kwargs...) =
	_χ²_loss(spectra_interp(total_model(tel, star), d.lsf), d; kwargs...)
function _loss_diagnostic(o::Output, om::OrderModel, d::Data;
	tel=nothing, star=nothing, rv=nothing, doppler_basis=nothing, kwargs...)
    !isnothing(tel) ? tel_o = spectra_interp(_eval_lm_vec(om, tel; log_lm=log_lm(om.tel.lm)), om.t2o) : tel_o = o.tel
	if typeof(om) <: OrderModelDPCA
		!isnothing(star) ? star_o = spectra_interp(_eval_lm_vec(om, star; log_lm=log_lm(om.star.lm)), om.b2o) : star_o = o.star
		if !isnothing(doppler_basis) && !isnothing(rv)
			rv_o = spectra_interp(doppler_basis * rv, om.b2o)
		elseif !isnothing(rv)
			rv_o = spectra_interp(_eval_lm(om.rv.lm.M, rv), om.b2o)
		else
			rv_o = o.rv
		end
		return __loss_diagnostic(tel_o, star_o, rv_o, d; kwargs...)
	end
	if !isnothing(star)
		if !isnothing(rv)
			star_o = spectra_interp(_eval_lm_vec(om, star; log_lm=log_lm(om.star.lm)), rv .+ om.bary_rvs, om.b2o)
		else
			star_o = spectra_interp(_eval_lm_vec(om, star; log_lm=log_lm(om.star.lm)), om.rv .+ om.bary_rvs, om.b2o)
		end
	elseif !isnothing(rv)
		star_o = spectra_interp(om.star.lm(), rv .+ om.bary_rvs, om.b2o)
	else
		star_o = o.star
	end
	return __loss_diagnostic(tel_o, star_o, d; kwargs...)
end
_loss_diagnostic(mws::ModelWorkspace; kwargs...) = _loss_diagnostic(mws.o, mws.om, mws.d; kwargs...)

# summed χ² loss functions
_loss(tel, star, rv, d::Data; kwargs...) = sum(__loss_diagnostic(tel, star, rv, d; kwargs...))
_loss(tel, star, d::Data; kwargs...) = sum(__loss_diagnostic(tel, star, d; kwargs...))
_loss(o::Output, om::OrderModel, d::Data; kwargs...) = sum(_loss_diagnostic(o, om, d; kwargs...))
_loss(mws::ModelWorkspace; kwargs...) = _loss(mws.o, mws.om, mws.d; kwargs...)


# Extract the star template μ from the `star` kwarg passed to loss functions.
# The kwarg follows _lm_tuple / vec(lm) conventions:
#   TemplateModel  → (μ,) or [μ]         → first element
#   FullLinearModel → (M, s, μ) or [M,s,μ] → third element
#   BaseLinearModel → (M, s) or [M,s]     → no μ, fall back to `fallback`
_get_star_μ(::Nothing, fallback) = fallback
_get_star_μ(t::Tuple{<:Any}, fallback) = t[1]
_get_star_μ(t::Tuple{<:Any,<:Any,<:Any}, fallback) = t[3]
_get_star_μ(t::Tuple{<:Any,<:Any}, fallback) = fallback
_get_star_μ(v::AbstractVector, fallback) = length(v) == 3 ? v[3] : (length(v) == 1 ? v[1] : fallback)

"""
	_loss_recalc_rv_basis(o, om, d; kwargs...)

`_loss()` with a fresh Doppler basis computed from the current star template.
The basis is passed as a `doppler_basis` kwarg (no mutation of `om.rv.lm.M`),
so AD engines can differentiate correctly through μ → doppler basis → loss.
"""
function _loss_recalc_rv_basis(o::Output, om::OrderModel, d::Data; star=nothing, rv=nothing, kwargs...)
	star_μ = _get_star_μ(star, om.star.lm.μ)
	doppler_M = doppler_component_AD(om.star.λ, star_μ)
	return _loss(o, om, d; star=star, rv=rv, doppler_basis=doppler_M, kwargs...)
end
_loss_recalc_rv_basis(mws::ModelWorkspace; kwargs...) = _loss_recalc_rv_basis(mws.o, mws.om, mws.d; kwargs...)


"""
	loss_func(mws; include_priors=false)

Create a loss function for the model and data in `mws`
"""
function loss_func(mws::ModelWorkspace; include_priors::Bool=false)
	if include_priors
		return (; kwargs...) -> _loss(mws; kwargs...) + tel_prior(mws.om) + star_prior(mws.om)
	else
		return (; kwargs...) -> _loss(mws; kwargs...)
	end
end


"""
	loss_funcs_telstar(o, om, d)

Create loss functions for changing
	- the telluric and stellar templates, features, and scores
	- the telluric and stellar scores
	- the RVs

Used to fit scores efficiently with L-BFGS
"""
function loss_funcs_telstar(o::Output, om::OrderModel, d::Data)
    l_telstar(telstar; kwargs...) =
        _loss(o, om, d; tel=telstar[1], star=telstar[2], kwargs...) +
			tel_prior(telstar[1], om) + star_prior(telstar[2], om)
	is_star_time_variable = is_time_variable(om.star)
    function l_telstar_s(telstar_s)
		prior = 0.
		if is_time_variable(om.tel)
			tel = [om.tel.lm.M, telstar_s[1], om.tel.lm.μ]
			prior += model_s_prior(telstar_s[1], om.reg_tel)
			if is_star_time_variable
				star = [om.star.lm.M, telstar_s[2], om.star.lm.μ]
				prior += model_s_prior(telstar_s[2], om.reg_star)
			else
				star = nothing
			end
		elseif is_star_time_variable
			tel = nothing
			star = [om.star.lm.M, telstar_s[1], om.star.lm.μ]
			prior += model_s_prior(telstar_s[1], om.reg_star)
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star, use_var_s=true) + prior
    end

    l_rv(rv) = _loss(o, om, d; rv=rv, use_var_s=true)

    return l_telstar, l_telstar_s, l_rv
end
loss_funcs_telstar(mws::ModelWorkspace) = loss_funcs_telstar(mws.o, mws.om, mws.d)


"""
	loss_funcs_telstar_v2(o, om, d)

Wobble analog of the DPCA inline `l_total` pattern. Takes `θ = (tel_tuple, star_tuple)`
where each element is a `_lm_tuple` (typed Tuple of arrays), so Enzyme's
`EnzymeNestedCache` (`Const(l)` + `Duplicated(θ_copy, ∂θ)`) correctly tracks all
active variables without `set_runtime_activity` on the outer unflatten step.

For Wobble there is no separate Doppler basis; the stellar gradient flows through
`spectra_interp(stellar_model, om.rv .+ om.bary_rvs, om.b2o)` natively.
`om.rv` is captured as non-differentiable data (only the linear-model parameters in θ
are differentiated).
"""
function loss_funcs_telstar_v2(o::Output, om::OrderModelWobble, d::Data)
    # Use Val-dispatch instead of keyword-dispatch so Enzyme can statically track
    # activity through the return value (Core.kwcall breaks activity analysis).
    function _ev(v, log_val::Val)
        length(v) == 3 && return _eval_lm_inner(v[1], v[2], v[3], log_val)
        length(v) == 2 && return _eval_lm_inner(v[1], v[2], log_val)
        return v[1] * ones(om.n)'  # TemplateModel (length-1)
    end
    function l_telstar_v2(telstar)
        tel_lm  = telstar[1]
        star_lm = telstar[2]
        tel_o  = spectra_interp(_ev(tel_lm,  Val(log_lm(om.tel.lm))),  om.t2o)
        star_o = spectra_interp(_ev(star_lm, Val(log_lm(om.star.lm))), om.rv .+ om.bary_rvs, om.b2o)
        return sum(__loss_diagnostic(tel_o, star_o, d)) +
               tel_prior(tel_lm, om) + star_prior(star_lm, om)
    end
    return l_telstar_v2
end


"""
	loss_funcs_total(o, om, d)

Create loss functions for changing
	- the telluric and stellar templates, features, and scores
	- the telluric and stellar scores and RVs

Used to fit models with ADAM
"""
# Migration note (Nabla → Enzyme), DPCA path only:
#
# Nabla precomputed `doppler_M` outside the loss closure (in the outer Adam loop)
# and captured it as a constant. Gradient therefore only flowed through the stellar
# spectrum path: ∂loss/∂μ = ∂loss/∂star_o · ∂star_o/∂μ.
#
# Enzyme traces through `doppler_M = doppler_component_AD(om.star.λ, star_μ)`
# inside `l_total`, so the gradient also flows through the Doppler basis:
#   ∂loss/∂μ += ∂loss/∂rv_o · ∂rv_o/∂doppler_M · ∂doppler_M/∂μ
# where ∂doppler_M/∂μ = ∂(simple_derivative_AD(μ) .* (λ ./ dλdpix))/∂μ.
# This second term is physically correct: a change in the stellar template changes
# the log-wavelength derivative used as the Doppler basis vector.
#
# The Wobble path is unaffected (no Doppler basis derived from μ).
function loss_funcs_total(o::Output, om::OrderModelDPCA, d::Data)
    # Enzyme requires all active intermediates to flow through positional arguments,
    # not kwargs. `doppler_M` is freshly allocated inside `l_total` and has no entry
    # in Enzyme's shadow for θ; routing it through kwargs loses its activity
    # (Enzyme's runtime check only tracks pointers present in the original θ shadow).
    function l_total(total)
        tel_lm  = total[1]
        star_lm = total[2]
        rv_s    = total[3]
        star_μ    = _get_star_μ(star_lm, om.star.lm.μ)
        doppler_M = doppler_component_AD(om.star.λ, star_μ)
        tel_o  = spectra_interp(_eval_lm_vec(om, tel_lm;  log_lm=log_lm(om.tel.lm)),  om.t2o)
        star_o = spectra_interp(_eval_lm_vec(om, star_lm; log_lm=log_lm(om.star.lm)), om.b2o)
        rv_o   = spectra_interp(doppler_M .* rv_s, om.b2o)
        return sum(__loss_diagnostic(tel_o, star_o, rv_o, d)) +
               tel_prior(tel_lm, om) + star_prior(star_lm, om)
    end
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	_tel_log_lm = Val(log_lm(om.tel.lm))
	_star_log_lm = Val(log_lm(om.star.lm))
    function l_total_s(total_s)
		prior = 0.
		rv_s = total_s[1 + is_star_time_variable + is_tel_time_variable]
		if is_tel_time_variable
			s_tel = total_s[1]
			tel_o = spectra_interp(_eval_lm_inner(om.tel.lm.M, s_tel, om.tel.lm.μ, _tel_log_lm), om.t2o)
			prior += model_s_prior(s_tel, om.reg_tel)
		else
			tel_o = o.tel
		end
		if is_star_time_variable
			s_star = total_s[1 + is_tel_time_variable]
			star_o = spectra_interp(_eval_lm_inner(om.star.lm.M, s_star, om.star.lm.μ, _star_log_lm), om.b2o)
			prior += model_s_prior(s_star, om.reg_star)
		else
			star_o = o.star
		end
		rv_o = spectra_interp(om.rv.lm.M * rv_s, om.b2o)
		return sum(__loss_diagnostic(tel_o, star_o, rv_o, d; use_var_s=true)) + prior
    end

    return l_total, l_total_s
end
function loss_funcs_total(o::Output, om::OrderModelWobble, d::Data)
	l_total(total) =
		_loss(o, om, d; tel=total[1], star=total[2], rv=total[3]) +
		tel_prior(total[1], om) + star_prior(total[2], om)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	_tel_log_lm = Val(log_lm(om.tel.lm))
	_star_log_lm = Val(log_lm(om.star.lm))
	# Specialized l_total_s closures avoid Const/Active PHI nodes in the Enzyme trace.
	# A single closure with `if is_tel_time_variable` compiles both branches and inserts
	# a PHI merging Const o.tel with an Active spectra_interp result — even when the
	# Const branch is dead at runtime. Lifting the branch outside the closure body means
	# each anonymous function has an unconditional body; Enzyme sees no mixed PHI nodes.
	l_total_s = if is_tel_time_variable && is_star_time_variable
		function(total_s)
			s_tel = total_s[1]; s_star = total_s[2]; rv = total_s[3]
			tel_o = spectra_interp(_eval_lm_inner(om.tel.lm.M, s_tel, om.tel.lm.μ, _tel_log_lm), om.t2o)
			star_o = spectra_interp(_eval_lm_inner(om.star.lm.M, s_star, om.star.lm.μ, _star_log_lm), _rv_shift(rv, om.bary_rvs), om.b2o)
			sum(__loss_diagnostic(tel_o, star_o, d; use_var_s=true)) +
				model_s_prior(s_tel, om.reg_tel) + model_s_prior(s_star, om.reg_star)
		end
	elseif is_tel_time_variable
		_const_star_flux = om.star.lm()
		function(total_s)
			s_tel = total_s[1]; rv = total_s[2]
			tel_o = spectra_interp(_eval_lm_inner(om.tel.lm.M, s_tel, om.tel.lm.μ, _tel_log_lm), om.t2o)
			star_o = spectra_interp(_const_star_flux, _rv_shift(rv, om.bary_rvs), om.b2o)
			sum(__loss_diagnostic(tel_o, star_o, d; use_var_s=true)) + model_s_prior(s_tel, om.reg_tel)
		end
	elseif is_star_time_variable
		function(total_s)
			s_star = total_s[1]; rv = total_s[2]
			star_o = spectra_interp(_eval_lm_inner(om.star.lm.M, s_star, om.star.lm.μ, _star_log_lm), _rv_shift(rv, om.bary_rvs), om.b2o)
			sum(__loss_diagnostic(o.tel, star_o, d; use_var_s=true)) + model_s_prior(s_star, om.reg_star)
		end
	else
		_const_star_flux = om.star.lm()
		function(total_s)
			rv = total_s[1]
			star_o = spectra_interp(_const_star_flux, _rv_shift(rv, om.bary_rvs), om.b2o)
			sum(__loss_diagnostic(o.tel, star_o, d; use_var_s=true))
		end
	end
    return l_total, l_total_s
end
loss_funcs_total(mws::ModelWorkspace) = loss_funcs_total(mws.o, mws.om, mws.d)



"""
	loss_funcs_frozen_tel(o, om, d)

Create loss functions for changing
	- the stellar templates, features, and scores and telluric scores
	- the telluric and stellar scores and RVs

Used to fit models with a set telluric model
"""
function loss_funcs_frozen_tel(o::Output, om::OrderModel, d::Data)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	function l_frozen_tel(total)
		is_tel_time_variable ? tel = (om.tel.lm.M, total[1], om.tel.lm.μ) : tel = nothing
		star = total[1+is_tel_time_variable]
		rv = total[2+is_tel_time_variable]
		return _loss(o, om, d; tel=tel, star=star, rv=rv) + star_prior(total[1+is_tel_time_variable], om)
	end
    function l_frozen_tel_s(total_s)
		prior = 0.
		if is_tel_time_variable
			tel = (om.tel.lm.M, total_s[1], om.tel.lm.μ)
			if is_star_time_variable
				star = (om.star.lm.M, total_s[2], om.star.lm.μ)
				prior += model_s_prior(total_s[2], om.reg_star)
			else
				star = nothing
			end
		elseif is_star_time_variable
			tel = nothing
			star = (om.star.lm.M, total_s[1], om.star.lm.μ)
			prior += model_s_prior(total_s[1], om.reg_star)
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star, rv=total_s[1+is_star_time_variable+is_tel_time_variable], use_var_s=true) + prior
    end
    return l_frozen_tel, l_frozen_tel_s
end
loss_funcs_frozen_tel(mws::ModelWorkspace) = loss_funcs_frozen_tel(mws.o, mws.om, mws.d)


## ADAM things
α, β1, β2, ϵ = 2e-3, 0.9, 0.999, 1e-8

# Return a concretely-typed Tuple of the parameter arrays in a LinearModel.
# Using a Tuple (rather than a Vector) lets Mooncake infer concrete element types
# throughout the loss body, enabling rule dispatch and efficient compiled code.
_lm_tuple(lm::FullLinearModel) = (lm.M, lm.s, lm.μ)
_lm_tuple(lm::BaseLinearModel) = (lm.M, lm.s)
_lm_tuple(lm::TemplateModel) = (lm.μ,)


"""
	Adam

Holds information used for Adaptive Moment Estimation optimization
"""
mutable struct Adam{T<:AbstractArray}
	"Learning rate"
    α::Float64
	"Exponential decay rate for the first moment estimates"
    β1::Float64
	"Exponential decay rate for the second-moment estimates"
    β2::Float64
	"Holder for aggregate of gradients"
    m::T
	"Holder for sum of square of past gradients"
    v::T
	"Holder for accelerated β1"
    β1_acc::Float64
	"Holder for accelerated β2"
    β2_acc::Float64
	"A small positive constant"
    ϵ::Float64
end
Adam(θ0::AbstractArray, α::Float64, β1::Float64, β2::Float64, ϵ::Float64) =
	Adam(α, β1, β2, vector_zero(θ0), vector_zero(θ0), β1, β2, ϵ)
Adam(θ0::AbstractArray; α::Float64=α, β1::Float64=β1, β2::Float64=β2, ϵ::Float64=ϵ) =
	Adam(θ0, α, β1, β2, ϵ)
Adams(θ0s::Vector{<:AbstractArray}, α::Float64, β1::Float64, β2::Float64, ϵ::Float64) =
	Adams.(θ0s, α, β1, β2, ϵ)
Adams(θ0s; α::Float64=α, β1::Float64=β1, β2::Float64=β2, ϵ::Float64=ϵ) =
	Adams(θ0s, α, β1, β2, ϵ)
Adams(θ0::AbstractVecOrMat{<:Real}, α::Float64, β1::Float64, β2::Float64, ϵ::Float64) =
	Adam(θ0, α, β1, β2, ϵ)
Adams(θ0s::Tuple; α::Float64=α, β1::Float64=β1, β2::Float64=β2, ϵ::Float64=ϵ) =
	map(t -> Adams(t; α=α, β1=β1, β2=β2, ϵ=ϵ), θ0s)
Base.copy(opt::Adam) = Adam(opt.α, opt.β1, opt.β2, opt.m, opt.v, opt.β1_acc, opt.β2_acc, opt.ϵ)


"""
	AdamState

Holds diagonstic information on the current state of an ADAM opmitization
"""
mutable struct AdamState
	"Interation number"
    iter::Int
	"Loss"
    ℓ::Float64
	"L1 norm of the gradient"
    L1_Δ::Float64
	"L2 norm of the gradient"
    L2_Δ::Float64
	"L∞ norm of the gradient"
    L∞_Δ::Float64
	"Change in the loss between iterations"
	δ_ℓ::Float64
	"Change in the L1 norm between iterations"
	δ_L1_Δ::Float64
	"Change in the L2 norm between iterations"
	δ_L2_Δ::Float64
	"Change in the L∞ norm between iterations"
	δ_L∞_Δ::Float64
end
AdamState() = AdamState(0, 0., 0., 0., 0., 0., 0., 0., 0.)
function println(as::AdamState)
    # println("Iter:  ", as.iter)
    println("ℓ:     ", as.ℓ,    "  ℓ_$(as.iter)/ℓ_$(as.iter-1):       ", as.δ_ℓ)
	println("L2_Δ:  ", as.L2_Δ, "  L2_Δ_$(as.iter)/L2_Δ_$(as.iter-1): ", as.δ_L2_Δ)
	println()
end

function iterate!(θs::Vector{<:AbstractArray}, ∇θs::Vector{<:AbstractArray}, opts::Vector)
    @assert length(θs) == length(∇θs) == length(opts)
	@inbounds for i in eachindex(θs)
		iterate!(θs[i], ∇θs[i], opts[i])
    end
end
function iterate!(θs::Tuple, ∇θs, opts)
    @inbounds for i in eachindex(θs)
        iterate!(θs[i], ∇θs[i], opts[i])
    end
end


"""
	iterate!(θ, ∇θ, opt)

Perform an ADAM optimization step based on the contents of `opt` on `θ`
"""
function iterate!(θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam)
	α=opt.α; β1=opt.β1; β2=opt.β2; ϵ=opt.ϵ; β1_acc=opt.β1_acc; β2_acc=opt.β2_acc; m=opt.m; v=opt.v
    one_minus_β1 = 1.0 - β1
	one_minus_β2 = 1.0 - β2
	one_minus_β1_acc = 1 - β1_acc
	one_minus_β2_acc = 1 - β2_acc
	# the matrix and dotted version is slower
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + one_minus_β1 * ∇θ[n]
        v[n] = β2 * v[n] + one_minus_β2 * ∇θ[n]^2
        m̂ = m[n] / one_minus_β1_acc
        v̂ = v[n] / one_minus_β2_acc
        θ[n] -= α * m̂ / (sqrt(v̂) + ϵ)
    end
	β1_acc *= β1
	β2_acc *= β2
end


function AdamState!_helper(as::AdamState, f::Symbol, val)
	setfield!(as, Symbol(:δ_,f), val / getfield(as, f))
	setfield!(as, f, val)
end


"""
	AdamState!(as, ℓ, Δ)

Update `as` with the next iteration's loss and gradient
"""
function AdamState!(as::AdamState, ℓ, Δ)
	as.iter += 1
	AdamState!_helper(as, :ℓ, ℓ)
	flat_Δ = Iterators.flatten(Iterators.flatten(Δ))
	AdamState!_helper(as, :L1_Δ, L1(flat_Δ))
	AdamState!_helper(as, :L2_Δ, L2(flat_Δ))
	AdamState!_helper(as, :L∞_Δ, L∞(Δ))
end

_verbose_def = false
_iter_def = 100
_f_reltol_def = 1e-4
_g_reltol_def = 1e-3
_g_L∞tol_def = 1e3
_f_reltol_def_s = 0
_g_L∞tol_def_s = 1e-8


"""
	AdamSubWorkspace

Holds a set of model parameters and the ADAM optimizer and functions used to optimize them
"""
struct AdamSubWorkspace{T,C,L<:Function}
	"Model parameters to optimize"
	θ::T
	"Adam optimizer parameters"
	opt#::Adam
	"Optimization state"
	as::AdamState
	"Loss function"
	l::L
	"AD gradient cache"
	cache::C
	function AdamSubWorkspace(θ::T, opt, as, l::L, cache::C) where {T,C,L<:Function}
		@assert typeof(l(θ)) <: Real
		return new{T,C,L}(θ, opt, as, l, cache)
	end
end
function AdamSubWorkspace(θ, l::Function; backend::ADBackend=EnzymeBackend(), kwargs...)
	# kwargs (om, build_θ, build_l, o, d) carry aliasing info needed by the
	# EnzymeBackend nested-tuple path. Mooncake's prepare_gradient ignores them.
	cache = prepare_gradient(backend, l, θ; kwargs...)
	return AdamSubWorkspace(θ, Adams(θ), AdamState(), l, cache)
end


"""
	update!(aws; careful_first_step=true, speed_up=false)

Perform an ADAM optimization step for the model parameters in `aws`
"""
function update!(aws::AdamSubWorkspace; careful_first_step::Bool=true, speed_up::Bool=false)

    val, Δ = value_and_gradient!(aws.cache, aws.l, aws.θ)
	AdamState!(aws.as, val, Δ)

	# if you want to make sure the learning rate doesn't start too big
	if careful_first_step && aws.as.iter==1
		first_iterate!(aws.l, val, aws.θ, aws.θ, Δ, aws.opt)
	# if you want to make sure the learning rate isn't too small (much more dangerous)
	elseif speed_up && (aws.as.iter > 10 && aws.as.iter%20==5)
		speed_up_iterate!(aws.l, aws.θ, aws.θ, Δ, aws.opt)
	else
    	iterate!(aws.θ, Δ, aws.opt)
	end

end


"""
	first_iterate!(l, l0, θs, θ, ∇θ, opt; ind=[], verbose=false)

Perform an ADAM optimization step based on the contents of `opt` on `θ` and decreases the learning rate to ensure the loss actually decreases
"""
function first_iterate!(l::Function, l0::Real, θs, θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam; ind=[], verbose::Bool=false)
	β1=opt.β1; β2=opt.β2; ϵ=opt.ϵ; β1_acc=opt.β1_acc; β2_acc=opt.β2_acc; m=opt.m; v=opt.v
	one_minus_β1 = 1.0 - β1
	one_minus_β2 = 1.0 - β2
	one_minus_β1_acc = 1 - β1_acc
	one_minus_β2_acc = 1 - β2_acc
    # the matrix and dotted version is slower
	θ_step = Array{Float64}(undef, size(m))
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + one_minus_β1 * ∇θ[n]
        v[n] = β2 * v[n] + one_minus_β2 * ∇θ[n]^2
        m̂ = m[n] / one_minus_β1_acc
        v̂ = v[n] / one_minus_β2_acc
        θ_step[n] = m̂ / (sqrt(v̂) + ϵ)
    end
	β1_acc *= β1
	β2_acc *= β2
	θ .-= opt.α .* θ_step

	# keep reducing `opt.α` until `l1` is less than `l0`
	l1 = l(θs)
	factor = 1
	while l1 > l0
		factor *= 2 
		opt.α /= 2
		θ .+= opt.α * θ_step
		l1 = l(θs)
	end
	if verbose && factor > 1; println("shrunk α$ind by a factor of $factor") end
end
function first_iterate!(l::Function, l0::Real, θs_unchanging::Vector{<:AbstractArray}, θs::Vector{<:AbstractArray}, ∇θs::Vector{<:AbstractArray}, opts::Vector; ind=Int[], kwargs...)
    @assert length(θs) == length(∇θs) == length(opts)
	@inbounds for i in eachindex(θs)
		first_iterate!(l, l0, θs_unchanging, θs[i], ∇θs[i], opts[i]; ind=append!(copy(ind),[i]), kwargs...)
    end
end
function first_iterate!(l::Function, l0::Real, θs_unchanging::Tuple, θs::Tuple, ∇θs, opts; ind=Int[], kwargs...)
    @inbounds for i in eachindex(θs)
        first_iterate!(l, l0, θs_unchanging, θs[i], ∇θs[i], opts[i]; ind=append!(copy(ind),[i]), kwargs...)
    end
end



"""
	speed_up_iterate!(l, θs, θ, ∇θ, opt; ind=[], verbose=false)

Perform an ADAM optimization step based on the contents of `opt` on `θ` and increases the learning rate to attempt to speed up the optimization
"""
function speed_up_iterate!(l::Function, θs, θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam; ind=[], verbose::Bool=false)
	β1=opt.β1; β2=opt.β2; ϵ=opt.ϵ; β1_acc=opt.β1_acc; β2_acc=opt.β2_acc; m=opt.m; v=opt.v
	one_minus_β1 = 1.0 - β1
	one_minus_β2 = 1.0 - β2
	one_minus_β1_acc = 1 - β1_acc
	one_minus_β2_acc = 1 - β2_acc
    # the matrix and dotted version is slower
	θ_step = Array{Float64}(undef, size(m))
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + one_minus_β1 * ∇θ[n]
        v[n] = β2 * v[n] + one_minus_β2 * ∇θ[n]^2
        m̂ = m[n] / one_minus_β1_acc
        v̂ = v[n] / one_minus_β2_acc
        θ_step[n] = m̂ / (sqrt(v̂) + ϵ)
    end
	β1_acc *= β1
	β2_acc *= β2
	θ .-= opt.α .* θ_step
	l1 = l(θs)
	θ .-= opt.α .* θ_step

	# keep increasing `opt.α` while `l2` is still lower than `l1`
	l2 = l(θs)
	factor = 1
	while l1 > l2
		factor *= 2
		opt.α *= 2
		θ .-= opt.α * θ_step
		l1 = l2
		l2 = l(θs)
	end
	θ .+= opt.α .* θ_step
	if verbose && factor > 1; println("increased α$ind by a factor of $factor") end
end
function speed_up_iterate!(l::Function, θs_unchanging::Vector{<:AbstractArray}, θs::Vector{<:AbstractArray}, ∇θs::Vector{<:AbstractArray}, opts::Vector; ind=Int[], kwargs...)
    @assert length(θs) == length(∇θs) == length(opts)
	@inbounds for i in eachindex(θs)
		speed_up_iterate!(l, θs_unchanging, θs[i], ∇θs[i], opts[i]; ind=append!(copy(ind),[i]), kwargs...)
    end
end
function speed_up_iterate!(l::Function, θs_unchanging::Tuple, θs::Tuple, ∇θs, opts; ind=Int[], kwargs...)
    @inbounds for i in eachindex(θs)
        speed_up_iterate!(l, θs_unchanging, θs[i], ∇θs[i], opts[i]; ind=append!(copy(ind),[i]), kwargs...)
    end
end


"""
	check_converged(as, f_reltol, g_reltol, g_L∞tol)

Check to see if the Adam optimization has coverged based on the change in loss or its gradient
"""
function check_converged(as::AdamState, f_reltol::Real, g_reltol::Real, g_L∞tol::Real)
	as.ℓ > 0 ? δ_ℓ = as.δ_ℓ : δ_ℓ = 1 / as.δ_ℓ  # further reductions in negative cost functions are good!
	return ((δ_ℓ > (1 - f_reltol)) && (max(as.δ_L2_Δ,1/as.δ_L2_Δ) < (1+abs(g_reltol)))) || (as.L∞_Δ < g_L∞tol)
end
check_converged(as::AdamState, iter::Int, f_reltol::Real, g_reltol::Real, g_L∞tol::Real) = (as.iter > iter) || check_converged(as, f_reltol, g_reltol, g_L∞tol)


"""
	train_SubModel!(aws; iter=_iter_def, f_reltol=_f_reltol_def, g_reltol=_g_reltol_def, g_L∞tol=_g_L∞tol_def, cb=(as::AdamState)->(), careful_first_step=true, speed_up=false, kwargs...)

Train the model parameters in `aws` for up to `iter` Adam iterations until it converges based on `check_converged()`
"""
function train_SubModel!(aws::AdamSubWorkspace; iter=_iter_def, f_reltol = _f_reltol_def, g_reltol = _g_reltol_def, g_L∞tol = _g_L∞tol_def, cb::Function=(as::AdamState)->(), careful_first_step::Bool=true, speed_up::Bool=false, kwargs...)
	converged = false  # check_converged(aws.as, iter, f_tol, g_tol)
	while !converged
		update!(aws; careful_first_step=careful_first_step, speed_up=speed_up)
		cb(aws.as)
		converged = check_converged(aws.as, iter, f_reltol, g_reltol, g_L∞tol)
	end
	converged = check_converged(aws.as, f_reltol, g_reltol, g_L∞tol)
	# converged ? println("Converged") : println("Max Iters Reached")
	return converged
end


"""
	scale_α_helper!(opt, α_ratio, θ, α, scale_α)

Optionally scale `opt.α` based on the amplitudes in `θ`
"""
function scale_α_helper!(opt::Adam, α_ratio::Real, θ::AbstractVecOrMat, α::Real, scale_α::Bool)
	scale_α ? opt.α = α_ratio * rel_step_size(θ) : opt.α = α
end
function scale_α_helper!(opts::Vector, α_ratio::Real, θs, α::Real, scale_α::Bool)
	@inbounds for i in eachindex(opts)
		scale_α_helper!(opts[i], α_ratio, θs[i], α, scale_α)
	end
end
function scale_α_helper!(opts::Tuple, α_ratio::Real, θs, α::Real, scale_α::Bool)
    @inbounds for i in eachindex(opts)
        scale_α_helper!(opts[i], α_ratio, θs[i], α, scale_α)
    end
end
rel_step_size(θ::AbstractVecOrMat) = sqrt(mean(abs2, θ))
_scale_α_def = false


"""
	TotalWorkspace

A workspace to optimize all of the parameters in a SSOF model and data it is to be optimized on
Uses our custom implementation of ADAM
"""
struct TotalWorkspace <: AdamWorkspace
	"Optimization workspace for all model parameters"
	total::AdamSubWorkspace
	"SSOF model"
	om::OrderModel
	"SSOF model output"
	o::Output
	"pre-formatted SSOF data"
	d::Data
	"Whether or not the templates and features should be fit"
	only_s::Bool
end

function TotalWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def, kwargs...)
	l_total, l_total_s = loss_funcs_total(o, om, d)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	# Builders capture the four-way (only_s, is_tel_time_variable, is_star_time_variable)
	# branching so the EnzymeBackend nested-tuple path can rebuild a shadow θ
	# matching the primal's aliasing structure into ∂om. Mooncake ignores them.
	rvs_of(om2) = typeof(om2) <: OrderModelDPCA ? om2.rv.lm.s : om2.rv
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				build_θ = om2 -> (om2.tel.lm.s, om2.star.lm.s, rvs_of(om2))
			else
				build_θ = om2 -> (om2.tel.lm.s, rvs_of(om2))
			end
		elseif is_star_time_variable
			build_θ = om2 -> (om2.star.lm.s, rvs_of(om2))
		else
			build_θ = om2 -> (rvs_of(om2),)
		end
		loss = l_total_s
		build_l = (om2, o2, d2) -> loss_funcs_total(o2, om2, d2)[2]
	else
		build_θ = om2 -> (_lm_tuple(om2.tel.lm), _lm_tuple(om2.star.lm), rvs_of(om2))
		loss = l_total
		build_l = (om2, o2, d2) -> loss_funcs_total(o2, om2, d2)[1]
	end
	total = AdamSubWorkspace(build_θ(om), loss; om=om, build_θ=build_θ, build_l=build_l, o=o, d=d, kwargs...)
	if is_tel_time_variable || is_star_time_variable
		scale_α_helper!(total.opt[1:(is_tel_time_variable+is_star_time_variable)], α_ratio, total.θ, α, scale_α)
	end
	scale_α_helper!(total.opt[end], α_ratio, total.θ[end], α, true)
	return TotalWorkspace(total, om, o, d, only_s)
end
TotalWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	TotalWorkspace(om(inds), d(inds); kwargs...)
TotalWorkspace(om::OrderModel, d::Data; kwargs...) =
	TotalWorkspace(Output(om, d), om, d; kwargs...)
Base.copy(mws::TotalWorkspace) = TotalWorkspace(copy(mws.om), mws.d)


"""
	FrozenTelWorkspace

A workspace to optimize all of the parameters in a SSOF model (except for the tellruc template and features) and data it is to be optimized on
"""
struct FrozenTelWorkspace <: AdamWorkspace
	"Optimization workspace for (nearly) all model parameters"
	total::AdamSubWorkspace
	"SSOF model"
	om::OrderModel
	"SSOF model output"
	o::Output
	"pre-formatted SSOF data"
	d::Data
	"Whether or not the templates and features should be fit"
	only_s::Bool
end


function FrozenTelWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def, kwargs...)
	l_frozen_tel, l_frozen_tel_s = loss_funcs_frozen_tel(o, om, d)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	# See TotalWorkspace for the builder rationale.
	rvs_of(om2) = typeof(om2) <: OrderModelDPCA ? om2.rv.lm.s : om2.rv
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				build_θ = om2 -> (om2.tel.lm.s, om2.star.lm.s, rvs_of(om2))
			else
				build_θ = om2 -> (om2.tel.lm.s, rvs_of(om2))
			end
		elseif is_star_time_variable
			build_θ = om2 -> (om2.star.lm.s, rvs_of(om2))
		else
			build_θ = om2 -> (rvs_of(om2),)
		end
		loss = l_frozen_tel_s
		build_l = (om2, o2, d2) -> loss_funcs_frozen_tel(o2, om2, d2)[2]
	else
		if is_tel_time_variable
			build_θ = om2 -> (om2.tel.lm.s, _lm_tuple(om2.star.lm), rvs_of(om2))
		else
			build_θ = om2 -> (_lm_tuple(om2.star.lm), rvs_of(om2))
		end
		loss = l_frozen_tel
		build_l = (om2, o2, d2) -> loss_funcs_frozen_tel(o2, om2, d2)[1]
	end
	total = AdamSubWorkspace(build_θ(om), loss; om=om, build_θ=build_θ, build_l=build_l, o=o, d=d, kwargs...)
	if is_tel_time_variable || is_star_time_variable
		scale_α_helper!(total.opt[1:(is_tel_time_variable+is_star_time_variable)], α_ratio, total.θ, α, scale_α)
	end
	scale_α_helper!(total.opt[end], α_ratio, total.θ[end], α, true)
	rm_dict!(om.reg_tel)
	return FrozenTelWorkspace(total, om, o, d, only_s)
end
FrozenTelWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	FrozenTelWorkspace(om(inds), d(inds); kwargs...)
FrozenTelWorkspace(om::OrderModel, d::Data; kwargs...) =
	FrozenTelWorkspace(Output(om, d), om, d; kwargs...)


"""
	ModelWorkspace(model, data)

Create a workspace for optimizing `model` with `data`
Creates a FrozenTelWorkspace if the model has no telluric feature vectors and an empty telluric template
"""
function ModelWorkspace(model::OrderModel, data::Data; kwargs...)
	if no_tellurics(model)
		return FrozenTelWorkspace(model, data; kwargs...)
	else
		return TotalWorkspace(model, data; kwargs...)
	end
end


"""
	train_OrderModel!(mws; ignore_regularization=false, verbose=_verbose_def, shift_scores=true, μ_positive=true, tel_μ_lt1=false, rm_doppler=true, kwargs...)

Train the model in `mws` with some optional modifications to the optimization (ignore_regularization, shift_score, μ_positive, tel_μ_lt1, rm_doppler)
"""
function train_OrderModel!(mws::AdamWorkspace; ignore_regularization::Bool=false, verbose::Bool=_verbose_def, shift_scores::Bool=true, μ_positive::Bool=true, tel_μ_lt1::Bool=false, rm_doppler::Bool=true, kwargs...)

	if rm_doppler; dop_comp_holder = Array{Float64}(undef, length(mws.om.star.lm.μ)) end

	update_interpolation_locations!(mws)

	# optionally ignore the regularization in `mws.om`
    if ignore_regularization
        reg_tel_holder = copy(mws.om.reg_tel)
        reg_star_holder = copy(mws.om.reg_star)
        rm_regularization!(mws.om)
    end

	# create a callback function that runs after every iteration
	function cb(as::AdamState)

		# optionally shift the score means to be near 0
		if shift_scores
			if !(typeof(mws) <: FrozenTelWorkspace)
				remove_lm_score_means!(mws.om.tel.lm; prop=0.2)
			end
			if typeof(mws.om) <: OrderModelWobble
				remove_lm_score_means!(mws.om.star.lm; prop=0.2)
			end
		end

		# optionally make the templates always positive
		if μ_positive
			mws.om.tel.lm.μ[mws.om.tel.lm.μ .< 1e-10] .= 1e-10
			mws.om.star.lm.μ[mws.om.star.lm.μ .< 1e-10] .= 1e-10
		end

		# optionally make the telluric template always less than 1
		if tel_μ_lt1
			mws.om.tel.lm.μ[mws.om.tel.lm.μ .> 1] .= 1
		end

		# optionally make the stellar feature vectors orthagonal to a doppler shift
		if rm_doppler && is_time_variable(mws.om.star.lm)
			if mws.om.star.lm.log
				dop_comp_holder[:] = doppler_component_log(mws.om.star.λ, mws.om.star.lm.μ)
			else
				dop_comp_holder[:] = doppler_component(mws.om.star.λ, mws.om.star.lm.μ)
			end
			for i in axes(mws.om.star.lm.M, 2)
				EMPCA._reorthogonalize_no_renorm!(view(mws.om.star.lm.M, :, i), dop_comp_holder)
			end
		end

		# keep the stored doppler basis current; _loss_recalc_rv_basis no longer
		# mutates om.rv.lm.M, so refresh it here for downstream consumers
		# (rv_model, Output!, and score-only loss paths read om.rv.lm.M directly)
		if typeof(mws.om) <: OrderModelDPCA
			mws.om.rv.lm.M[:, 1] .= doppler_component(mws.om.star.λ, mws.om.star.lm.μ)
		end

		# make sure the interpolation locations are still correct
		if as.iter % 10 == 9
			update_interpolation_locations!(mws)
		end

		if verbose; println(as) end
	end

	# train the model and update `mws.o`
	result = train_SubModel!(mws.total; cb=cb, kwargs...)
	mws.total.as.iter = 0
    Output!(mws)

	# reset the regularization
    if ignore_regularization
        copy_dict!(mws.om.reg_tel, reg_tel_holder)
        copy_dict!(mws.om.reg_star, reg_star_holder)
    end

	return result
end

Output!(mws::ModelWorkspace) = Output!(mws.o, mws.om, mws.d)


## Optim Versions


"""
	opt_funcs(loss, pars)

Create an objective object for Optim from `loss` that uses a flattened verison of `pars`
"""
function opt_funcs(loss::Function, pars::AbstractVecOrMat; backend::ADBackend=EnzymeBackend())
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns Vector of untransformed params
    f = FlatLoss(loss, unflatten)
	cache = prepare_gradient(backend, f, flat_initial_params)
    function g!(G, θ)
        G .= value_and_gradient!(cache, f, θ)[2]
    end
    function fg_obj!(G, θ)
		val, ∂θ = value_and_gradient!(cache, f, θ)
		G .= ∂θ
        return val
    end
    return flat_initial_params, OnceDifferentiable(f, g!, fg_obj!, flat_initial_params), unflatten
end


"""
	OptimSubWorkspace

Holds a set of model parameters and the Optim optimizer and functions used to optimize them
"""
struct OptimSubWorkspace
	"Model parameters to optimize"
    θ::AbstractVecOrMat
	"Optim objective object"
    obj::OnceDifferentiable
	"The first order optimizer to use"
    opt::Optim.FirstOrderOptimizer
	"Flattened version of `θ`"
    p0::Vector
	"Function to convert `p0` to `θ`"
    unflatten::Union{Function,DataType}
end
function OptimSubWorkspace(θ::AbstractVecOrMat, loss::Function; use_cg::Bool=true)
	p0, obj, unflatten = opt_funcs(loss, θ)
	# opt = LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=NaN))
	# use_cg ? opt = ConjugateGradient() : opt = LBFGS()
	opt = LBFGS()
	# initial_state(method::LBFGS, ...) doesn't use the options for anything
	return OptimSubWorkspace(θ, obj, opt, p0, unflatten)
end


"""
	OptimTelStarWorkspace

A workspace to go back and forth optimizing the telluric and stellar parameters then the RVs in a SSOF model and data it is to be optimized on
Uses methods in Optim
"""
struct OptimTelStarWorkspace <: OptimWorkspace
	"Optimization workspace for stellar and telluric model parameters"
    telstar::OptimSubWorkspace
	"Optimization workspace for RV model parameters"
    rv::OptimSubWorkspace
	"SSOF model"
    om::OrderModel
	"SSOF model output"
    o::Output
	"pre-formatted SSOF data"
    d::Data
	"Whether or not the templates and features should be fit"
    only_s::Bool
end
function OptimTelStarWorkspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
	loss_telstar, loss_telstar_s, loss_rv = loss_funcs_telstar(o, om, d)
	typeof(om) <: OrderModelDPCA ?
		rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true) :
		rv = OptimSubWorkspace(om.rv, loss_rv; use_cg=true)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				telstar = OptimSubWorkspace([om.tel.lm.s, om.star.lm.s], loss_telstar_s; use_cg=!only_s)
			else
				telstar = OptimSubWorkspace([om.tel.lm.s], loss_telstar_s; use_cg=!only_s)
			end
		elseif is_star_time_variable
			telstar = OptimSubWorkspace([om.star.lm.s], loss_telstar_s; use_cg=!only_s)
		else
			@error "This model has no time variability, so a workspace that only changes scores makes no sense"
		end
	else
		telstar = OptimSubWorkspace([vec(om.tel.lm), vec(om.star.lm)], loss_telstar; use_cg=!only_s)
	end
	return OptimTelStarWorkspace(telstar, rv, om, o, d, only_s)
end
OptimTelStarWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	OptimTelStarWorkspace(om(inds), d(inds); kwargs...)
OptimTelStarWorkspace(om::OrderModel, d::Data; kwargs...) =
	OptimTelStarWorkspace(om, Output(om, d), d; kwargs...)


"""
	OptimTotalWorkspace

A workspace to optimize all of the parameters in a SSOF model and data it is to be optimized on
Uses methods in Optim
"""
struct OptimTotalWorkspace <: OptimWorkspace
	"Optimization workspace for all model parameters"
    total::OptimSubWorkspace
	"SSOF model"
    om::OrderModel
	"SSOF model output"
    o::Output
	"pre-formatted SSOF data"
    d::Data
	"Whether or not the templates and features should be fit"
    only_s::Bool
end
function OptimTotalWorkspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
	l_total, l_total_s = loss_funcs_total(o, om, d)
	typeof(om) <: OrderModelDPCA ? rvs = om.rv.lm.s : rvs = om.rv
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				total = OptimSubWorkspace([om.tel.lm.s, om.star.lm.s, rvs], l_total_s; use_cg=true)
			else
				total = OptimSubWorkspace([om.tel.lm.s, rvs], l_total_s; use_cg=true)
			end
		elseif is_star_time_variable
			total = OptimSubWorkspace([om.star.lm.s, rvs], l_total_s; use_cg=true)
		else
			total = OptimSubWorkspace([rvs], l_total_s; use_cg=true)
		end
	else
		total = OptimSubWorkspace([vec(om.tel.lm), vec(om.star.lm), rvs], l_total)
	end
	return OptimTotalWorkspace(total, om, o, d, only_s)
end
OptimTotalWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	OptimTotalWorkspace(om(inds), d(inds); kwargs...)
OptimTotalWorkspace(om::OrderModel, d::Data; kwargs...) =
	OptimTotalWorkspace(om, Output(om, d), d; kwargs...)


"""
	_OSW_optimize!(osw, options)

Optimize the model in `osw`
"""
function _OSW_optimize!(osw::OptimSubWorkspace, options::Optim.Options)
    result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)
    osw.p0[:] = result.minimizer
    return result
end

function optim_print(x)
	println()
	println("ℓ:     ", x.f_x)
	println("L∞(∇): ", maximum(abs, x.g_x))
	println()
	# ends optimization if true
	return false
end

function optim_cb_f(; verbose::Bool=true)
    if verbose
		return x -> optim_print(x)
    else
		return x -> false
    end
end


"""
	train_OrderModel!(ow::OptimTelStarWorkspace; verbose=_verbose_def, iter=_iter_def, f_tol=_f_reltol_def, g_tol=_g_L∞tol_def, train_telstar=true, ignore_regularization=false, μ_positive=false, careful_first_step=true, speed_up=false, kwargs...)

Train the model in `ow`, training the telluric and stellar parameters, then the RVs
"""
function train_OrderModel!(ow::OptimTelStarWorkspace; verbose::Bool=_verbose_def, iter::Int=_iter_def, f_tol::Real=_f_reltol_def, g_tol::Real=_g_L∞tol_def, train_telstar::Bool=true, ignore_regularization::Bool=false, μ_positive::Bool=false, careful_first_step::Bool=true, speed_up::Bool=false, kwargs...)
    # `μ_positive`, `careful_first_step`, and `speed_up` are only included to prevent errors and do nothing
	
	optim_cb = optim_cb_f(; verbose=verbose)

	# optionally ignore the regularization in `ow.om`
    if ignore_regularization
        reg_tel_holder = copy(ow.om.reg_tel)
        reg_star_holder = copy(ow.om.reg_star)
        rm_regularization!(ow.om)
    end

	# train the telluric and stellar parameters if desired
    if train_telstar
        options = Optim.Options(;iterations=iter, f_reltol=f_tol, g_tol=g_tol, callback=optim_cb, kwargs...)
        # optimize tellurics and star
        result_telstar = _OSW_optimize!(ow.telstar, options)
		lm_vec = ow.telstar.unflatten(ow.telstar.p0)
        if ow.only_s
			if is_time_variable(ow.om.tel)
				ow.om.tel.lm.s[:] = lm_vec[1]
				if is_time_variable(ow.om.star)
					ow.om.star.lm.s[:] = lm_vec[2]
				end
			else
				ow.om.star.lm.s[:] = lm_vec[1]
			end
        else
            copy_to_LinearModel!(ow.om.tel.lm, lm_vec[1])
			copy_to_LinearModel!(ow.om.star.lm, lm_vec[2])
        end
        ow.o.star .= star_model(ow.om)
        ow.o.tel .= tel_model(ow.om)
    end

    # optimize RVs
	result_rv = train_rvs_optim!(ow, optim_cb; f_tol=f_tol, g_tol=g_tol, kwargs...)
    if typeof(ow.om) <: OrderModelDPCA
		ow.o.rv .= rv_model(ow.om)
	else
		ow.o.star .= star_model(ow.om)
	end

	recalc_total!(ow.o, ow.d)
    if ignore_regularization
        copy_dict!(ow.om.reg_tel, reg_tel_holder)
        copy_dict!(ow.om.reg_star, reg_star_holder)
    end
    return result_telstar, result_rv
end


"""
	train_OrderModel!(ow::OptimTelStarWorkspace; verbose=_verbose_def, iter=_iter_def, f_tol=_f_reltol_def, g_tol=_g_L∞tol_def, train_telstar=true, ignore_regularization=false, μ_positive=false, careful_first_step=true, speed_up=false, kwargs...)

Train the model in `ow`, training the telluric and stellar parameters, then the RVs
"""
function train_OrderModel!(ow::OptimTotalWorkspace; verbose::Bool=_verbose_def, iter::Int=_iter_def, f_tol::Real=_f_reltol_def, g_tol::Real=_g_L∞tol_def, ignore_regularization::Bool=false, μ_positive::Bool=false, careful_first_step::Bool=true, speed_up::Bool=false, kwargs...)
    # `μ_positive`, `careful_first_step`, and `speed_up` are only included to prevent errors and do nothing
	
	optim_cb = optim_cb_f(; verbose=verbose)

	# optionally ignore the regularization in `ow.om`
    if ignore_regularization
        reg_tel_holder = copy(ow.om.reg_tel)
        reg_star_holder = copy(ow.om.reg_star)
        rm_regularization!(ow.om)
    end

    options = Optim.Options(;iterations=iter, f_reltol=f_tol, g_tol=g_tol, callback=optim_cb, kwargs...)
    result_total = _OSW_optimize!(ow.total, options)
	lm_vec = ow.total.unflatten(ow.total.p0)
	is_tel_time_variable = is_time_variable(ow.om.tel)
	is_star_time_variable = is_time_variable(ow.om.star)
    if ow.only_s
		if is_tel_time_variable
			ow.om.tel.lm.s[:] = lm_vec[1]
			if is_star_time_variable
				ow.om.star.lm.s[:] = lm_vec[2]
			end
		else
			ow.om.star.lm.s[:] = lm_vec[1]
		end
		if typeof(ow.om) <: OrderModelDPCA
			ow.om.rv.lm.s[:] = lm_vec[1+is_tel_time_variable+is_star_time_variable]
		else
			ow.om.rv[:] = lm_vec[1+is_tel_time_variable+is_star_time_variable]
		end
    else
        copy_to_LinearModel!(ow.om.tel.lm, lm_vec[1])
		copy_to_LinearModel!(ow.om.star.lm, lm_vec[2])
		if typeof(ow.om) <: OrderModelDPCA
			ow.om.rv.lm.s[:] = lm_vec[3]
		else
			ow.om.rv[:] = lm_vec[3]
		end
    end

	ow.o.tel .= tel_model(ow.om)
    ow.o.star .= star_model(ow.om)
	if typeof(ow.om) <: OrderModelDPCA; ow.o.rv .= rv_model(ow.om) end
	recalc_total!(ow.o, ow.d)

    if ignore_regularization
        copy_dict!(ow.om.reg_tel, reg_tel_holder)
        copy_dict!(ow.om.reg_star, reg_star_holder)
    end
    return result_total
end


"""
	train_rvs_optim!(rv_ws, rv, optim_cb; g_tol=_g_L∞tol_def_s, f_tol=_f_reltol_def_s, iter=_iter_def, ignore_regularization=false, μ_positive=false, kwargs...)

Train the RVs from the `rv_ws` with Optim
"""
function train_rvs_optim!(rv_ws::OptimSubWorkspace, rv::AbstractVector, optim_cb::Function; g_tol::Real=_g_L∞tol_def_s, f_tol::Real=_f_reltol_def_s, iter::Int=_iter_def, ignore_regularization::Bool=false, μ_positive::Bool=false, kwargs...)
	# `μ_positive` and `ignore_regularization` are only included to prevent errors and do nothing
	options = Optim.Options(; callback=optim_cb, g_tol=g_tol, f_reltol=f_tol, iterations=iter, kwargs...)
	result_rv = _OSW_optimize!(rv_ws, options)
	rv[:] = rv_ws.unflatten(rv_ws.p0)
	return result_rv
end
# same as above but for DPCA models
function train_rvs_optim!(rv_ws::OptimSubWorkspace, rv::Submodel, star::Submodel, optim_cb::Function; g_tol::Real=_g_L∞tol_def_s, f_tol::Real=_f_reltol_def_s, iter::Int=_iter_def, kwargs...)
	options = Optim.Options(; callback=optim_cb, g_tol=g_tol, f_reltol=f_tol, iterations=iter, kwargs...)
	rv.lm.M .= doppler_component(star.λ, star.lm.μ)
	result_rv = _OSW_optimize!(rv_ws, options)
	rv.lm.s[:] = rv_ws.unflatten(rv_ws.p0)
	return result_rv
end
train_rvs_optim!(ow::OptimTelStarWorkspace, optim_cb::Function; kwargs...) =
	typeof(ow.om) <: OrderModelDPCA ?
		train_rvs_optim!(ow.rv, ow.om.rv, ow.om.star, optim_cb; kwargs...) :
		train_rvs_optim!(ow.rv, ow.om.rv, optim_cb; kwargs...)


"""
	finalize_scores_setup(mws; verbose=_verbose_def, f_tol=_f_reltol_def_s, g_tol=_g_L∞tol_def_s, careful_first_step=true, speed_up=false, kwargs...)

Create a function that optimizes the model scores with Optim
"""
function finalize_scores_setup(mws::ModelWorkspace; verbose::Bool=_verbose_def, f_tol::Real=_f_reltol_def_s, g_tol::Real=_g_L∞tol_def_s, careful_first_step::Bool=true, speed_up::Bool=false, kwargs...)
	if is_time_variable(mws.om.tel) || is_time_variable(mws.om.star)
		mws_s = OptimTotalWorkspace(mws.om, mws.d; only_s=true)  # does not converge reliably
		# mws_s = OptimTelStarWorkspace(mws.om, mws.d; only_s=true)
		score_trainer() = train_OrderModel!(mws_s; verbose=verbose, f_tol=f_tol, g_tol=g_tol, kwargs...)
		return score_trainer
	end
	optim_cb=optim_cb_f(; verbose=verbose)
	loss_rv(rv) = _loss(mws; rv=rv, use_var_s=true)
	return _finalize_scores_setup(mws, mws.om, loss_rv, optim_cb; f_tol=f_tol, g_tol=g_tol, kwargs...)
end
function _finalize_scores_setup(mws::ModelWorkspace, om::OrderModelDPCA, loss_rv::Function, optim_cb::Function; kwargs...)
	rv_ws = OptimSubWorkspace(mws.om.rv.lm.s, loss_rv; use_cg=true)
	rv_trainer() = train_rvs_optim!(rv_ws, mws.om.rv, mws.om.star, optim_cb; kwargs...)
	return rv_trainer
end
function _finalize_scores_setup(mws::ModelWorkspace, om::OrderModelWobble, loss_rv::Function, optim_cb::Function; kwargs...)
	rv_ws = OptimSubWorkspace(mws.om.rv, loss_rv; use_cg=true)
	rv_trainer() = train_rvs_optim!(rv_ws, mws.om.rv, optim_cb; kwargs...)
	return rv_trainer
end


"""
	finalize_scores!(score_trainer, mws)

Run `score_trainer` and update the output in `mws`
"""
function finalize_scores!(score_trainer::Function, mws::ModelWorkspace)
	result = score_trainer()
	Output!(mws)
	return result
end


"""
	finalize_scores!(mws; kwargs...)

Optimize the scores in `mws`
"""
function finalize_scores!(mws::ModelWorkspace; kwargs...)
	score_trainer = finalize_scores_setup(mws; kwargs...)
	return finalize_scores!(score_trainer, mws)
end

is_time_variable(lm::LinearModel) = !(typeof(lm) <: TemplateModel)
is_time_variable(sm::Submodel) = is_time_variable(sm.lm)


"""
	update_interpolation_locations!(om, d; use_mean=false)

Make sure the linear iterpolation locations for going from the stellar model to the data are correct as the RVs change
"""
function update_interpolation_locations!(om::OrderModel, d::Data; use_mean::Bool=false)
	# TODO: do this for undersamp_interp_helper as well
	if typeof(om) <: OrderModelWobble
		if use_mean
			StellarInterpolationHelper!(om.b2o,
				om.star.log_λ,
				om.bary_rvs .+ mean(om.rv),
				d.log_λ_obs)
		else
			StellarInterpolationHelper!(om.b2o,
				om.star.log_λ,
				om.bary_rvs + om.rv,
				d.log_λ_obs)
		end
	end
end
update_interpolation_locations!(mws::ModelWorkspace; kwargs...) = update_interpolation_locations!(mws.om, mws.d; kwargs...)


"""
	improve_model!(mws; verbose=true, kwargs...)

Train the model in `mws` with an extra step to ensure we are at a local maximum for the scores and RVs
"""
function improve_model!(mws::ModelWorkspace; verbose::Bool=true, kwargs...)
	train_OrderModel!(mws; verbose=verbose, kwargs...)  # 120s
	results = finalize_scores!(mws; verbose=verbose, kwargs...)
	return results
end


"""
	improve_initial_model!(mws; careful_first_step=true, speed_up=false, kwargs...)

Train the model in `mws` with an extra step to ensure we are at a local maximum for the scores and RVs
Defaults to taking a careful first step
"""
improve_initial_model!(mws::ModelWorkspace; careful_first_step::Bool=true, speed_up::Bool=false, kwargs...) = improve_model!(mws; verbose=false, ignore_regularization=true, μ_positive=true, careful_first_step=careful_first_step, speed_up=speed_up, kwargs...)


"""
	calculate_initial_model(data; kwargs...)
	
Find a SSOF model for a given dataset, `data`.
Defaults to returning the AIC-minimum model
	
# Optional arguments
- `instrument::String="None"`: The name of the instrument(s) the data was taken from. For bookkeeping
- `desired_order::Int=0`: What order (if any) the data was taken from. For bookkeeping
- `star::String="None"`: The name of the star the data was taken from. For bookkeeping
- `times::AbstractVector=1:size(data.flux, 2)`: The list of times (in days). Used to calculate intra-night RMS
- `μ_min::Real=0`: Set the minimum flux value for the output of `make_template()`
- `μ_max::Real=Inf`: Set the maximum flux value for the output of `make_template()`
- `use_mean::Bool=true`: Whether to use the mean or median for `make_template()`
- `stop_early::Bool=false`: Whether to stop the model search the first time adding a component increases the AIC
- `remove_reciprocal_continuum::Bool=false`: Whether you should attempt to remove places where the telluric template and stellar template are opposing each other (i.e. where continuum goes up in one and down in the other)
- `return_full_path::Bool=false`: Whether to return all of the searched models and metrics
- `max_n_tel::Int=5`: The maximum amount of telluric feature vectors to look for
- `max_n_star::Int=5`: The maximum amount of stellar feature vectors to look for
- `use_all_comps::Bool=false`: Whether to use all feature vectors, regardless of AIC
- `careful_first_step::Bool=true`: Whether to shrink the learning rates until the loss improves on the first iteration
- `speed_up::Bool=false`: Whether to inflate the learning rates until the loss is no longer improving throughout the optimization
- `log_λ_gp_star::Real=1/SOAP_gp_params.λ`: The log λ lengthscale of the stellar regularization GP
- `log_λ_gp_tel::Real=1/LSF_gp_params.λ`: The log λ lengthscale of the telluric regularization GP
- `kwargs...`: kwargs passed to `OrderModel` constructor
"""
function calculate_initial_model(data::Data;
	instrument::String="None", desired_order::Int=0, star::String="None", times::AbstractVector=1:size(data.flux, 2),
	μ_min::Real=0, μ_max::Real=Inf, use_mean::Bool=true, stop_early::Bool=false,
	remove_reciprocal_continuum::Bool=false, return_full_path::Bool=false,
	max_n_tel::Int=5, max_n_star::Int=5, use_all_comps::Bool=false, careful_first_step::Bool=true, speed_up::Bool=false, 
	log_λ_gp_star::Real=1/SOAP_gp_params.λ, log_λ_gp_tel::Real=1/LSF_gp_params.λ, kwargs...)
	# TODO: Make this work for OrderModelDPCA

	# Get non-LSF version of `data`
	d = GenericData(data)

	@assert max_n_tel >= -1
	@assert max_n_star >= 0

	# which amounts of feature vectors to test
	test_n_comp_tel = -1:max_n_tel
	test_n_comp_star = 0:max_n_star

	# initialize metric and model storage
	aics = Inf .* ones(length(test_n_comp_tel), length(test_n_comp_star))
	ℓs = -copy(aics)
	bics = copy(aics)
	rv_stds = copy(aics)
	rv_stds_intra = copy(aics)
	oms = Array{OrderModelWobble}(undef, length(test_n_comp_tel), length(test_n_comp_star))
	
	logdet_Σ, n = ℓ_prereqs(d.var)
	comp2ind(n_tel::Int, n_star::Int) = (n_tel+2, n_star+1)  # converts number of components to storage matrix index
	n_obs = size(d.flux, 2)

	om = OrderModel(d; instrument=instrument, order=desired_order, star_str=star, n_comp_tel=max_n_tel, n_comp_star=max_n_star, log_λ_gp_star=log_λ_gp_star, log_λ_gp_tel=log_λ_gp_tel, kwargs...)
	
	# get the stellar model wavelengths in observed frame as a function of time
	star_log_λ_tel = _shift_log_λ_model(d.log_λ_obs, d.log_λ_star, om.star.log_λ)
	# get the telluric model wavelengths in stellar frame as a function of time
	tel_log_λ_star = _shift_log_λ_model(d.log_λ_star, d.log_λ_obs, om.tel.log_λ)

	flux_star = ones(length(om.star.log_λ), n_obs)
	vars_star = SOAP_gp_var .* ones(length(om.star.log_λ), n_obs)
	flux_tel = ones(length(om.tel.log_λ), n_obs)
	vars_tel = SOAP_gp_var .* ones(length(om.tel.log_λ), n_obs)


	"Find places where the telluric and stellar continuums are reciprocals of each other"
	function reciprocal_continuum_mask(continuum::AbstractVector, other_interpolated_continuum::AbstractVector; probe_depth::Real=0.02, return_cc::Bool=false)
		

		cc = (1 .- continuum) .* (1 .- other_interpolated_continuum)
		ccm = find_modes(-cc)

		# find places where the continuums are opposite in sign and changing a similar amount
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

	function remove_reciprocal_continuum!(om::OrderModel, flux_star_holder::AbstractMatrix, vars_star_holder::AbstractMatrix, flux_tel_holder::AbstractMatrix, vars_tel_holder::AbstractMatrix; use_stellar_continuum::Bool=true, kwargs...)

		lm_tel = om.tel.lm
		lm_star = om.star.lm

		# Calculate a Rassine-like continuum for the telluric template
		_, c_t, _ = calc_continuum(om.tel.λ, lm_tel.μ, ones(length(lm_tel.μ)) ./ 1000;
			min_R_factor=1, smoothing_half_width=0,
			stretch_factor=10., merging_threshold = 0.)

		# Calculate a Rassine-like continuum for the stellar template
		_, c_s, _ = calc_continuum(om.star.λ, lm_star.μ, ones(length(lm_star.μ)) ./ 1000;
			min_R_factor=1,
			stretch_factor=10., merging_threshold = 0.)

		# interpolate the stellar continuum to the observed frame
		flux_star_holder .= c_s
		vars_star_holder .= SOAP_gp_var
		_spectra_interp_gp!(flux_tel_holder, vars_tel_holder, om.tel.log_λ, flux_star_holder, vars_star_holder, star_log_λ_tel; gp_mean=1., λ_kernel=1/log_λ_gp_star)
		
		# find where the continuums are reciprocal and remove the difference
		m, cc = reciprocal_continuum_mask(c_t, flux_tel_holder; return_cc=true, kwargs...)
		use_stellar_continuum ?
			lm_tel.μ[m] .*= vec(mean(flux_tel_holder[m, :]; dims=2)) :
			lm_tel.μ[m] ./= c_t[m]
		did_anything = any(m)

		# interpolate the telluric continuum to the stellar frame
		flux_tel .= c_t
		vars_tel .= SOAP_gp_var
		_spectra_interp_gp!(flux_star_holder, vars_star_holder, om.star.log_λ, flux_tel_holder, vars_tel_holder, tel_log_λ_star; gp_mean=1., λ_kernel=1/log_λ_gp_star)
		
		# find where the continuums are reciprocal and remove the difference
		m, cc = reciprocal_continuum_mask(c_s, flux_star_holder; return_cc=true, kwargs...)
		use_stellar_continuum ?
			lm_star.μ[m] ./= c_s[m] :
			lm_star.μ[m] .*= vec(mean(flux_star_holder[m, :]; dims=2))
		did_anything = did_anything || any(m)

		return did_anything

	end

	# remove the score means and flip the feature vectors
	function nicer_model!(mws::ModelWorkspace)
		remove_lm_score_means!(mws.om)
		flip_feature_vectors!(mws.om)
		mws.om.metadata[:todo][:initialized] = true
		mws.om.metadata[:todo][:downsized] = true
		# copy_dict!(mws.om.reg_tel, default_reg_tel)
		# copy_dict!(mws.om.reg_star, default_reg_star)
	end

	# 
	function get_metrics!(mws::ModelWorkspace, i::Int, j::Int)
		
		# # could set very small regularizations beforehand if we wanted
		# for (k, v) in mws.om.reg_tel
		# 	mws.om.reg_tel[k] = min_reg
		# end
		# for (k, v) in mws.om.reg_star
		# 	mws.om.reg_star[k] = min_reg
		# end


		try
			# improve the model
			improve_initial_model!(mws; careful_first_step=careful_first_step, speed_up=speed_up, iter=50)

			# if there is an LSF, do some more fitting
			if mws.d != data
				mws = typeof(mws)(copy(mws.om), data)
				improve_initial_model!(mws; careful_first_step=careful_first_step, speed_up=speed_up, iter=30)
			end
			nicer_model!(mws)

			# calculate metrics
			k = total_length(mws)
			ℓs[i,j] = ℓ(_loss(mws), logdet_Σ, n)
			if isnan(ℓs[i,j]); ℓs[i,j] = -Inf end
			aics[i,j] = aic(k, ℓs[i,j])
			bics[i,j] = bic(k, ℓs[i,j], n)
			model_rvs = rvs(mws.om)
			rv_stds[i,j] = std(model_rvs)
			rv_stds_intra[i,j] = intra_night_std(model_rvs, times; show_warn=false)

			return mws.om
		catch err
			if isa(err, DomainError)
				println("hit a domain error while optimizing")
				nicer_model!(mws)
				return mws.om
			else
				rethrow()
			end
		end
	end

	# stellar template model assuming no tellurics
	n_tel_cur = -1
	n_star_cur = 0
	search_new_tel = n_tel_cur+1 <= max_n_tel
	search_new_star = n_star_cur+1 <= max_n_star
	oms[1,1] = downsize(om, 0, 0)
	oms[1,1].tel.lm.μ .= 1
	_spectra_interp_gp!(flux_star, vars_star, oms[1,1].star.log_λ, d.flux, d.var .+ SOAP_gp_var, d.log_λ_star; gp_mean=1., λ_kernel=1/log_λ_gp_star)
	flux_star_no_tel = copy(flux_star)
	vars_star_no_tel = copy(vars_star)
	oms[1,1].star.lm.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=use_mean)
	
	# how good is the stellar template at modeling each pixel
	dop_comp = doppler_component(oms[1,1].star.λ, oms[1,1].star.lm.μ)
	# project_doppler_comp!(mws.om.rv, flux_star_no_tel .- mws.om.star.lm.μ, dop_comp, 1 ./ vars_star)
	mask_low_pixels!(flux_star_no_tel, vars_star_no_tel)
	mask_high_pixels!(flux_star_no_tel, vars_star_no_tel)
	χ²_star = vec(sum(_χ²_loss(star_model(oms[1,1]), d); dims=2))  # TODO: could optimize before checking this
	# star_template_χ² = sum(χ²_star)

	# get aic for base, only stellar template model
	mws = FrozenTelWorkspace(oms[1,1], d)
	om_cur = get_metrics!(mws, 1, 1)

	# telluric template assuming no stellar (will be overwritten later)
	_om = downsize(om, 0, 0)
	_om.star.lm.μ .= 1
	_spectra_interp_gp!(flux_tel, vars_tel, _om.tel.log_λ, d.flux, d.var .+ SOAP_gp_var, d.log_λ_obs; gp_mean=1., λ_kernel=1/log_λ_gp_star)
	_om.tel.lm.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)
	
	# how good is the stellar telluric template at modeling each pixel
	χ²_tel = vec(sum(_χ²_loss(tel_model(_om), d); dims=2))  # TODO: could optimize before checking this

	om_add_star = om_cur

	# get `flux_to` in the desired frame after interpolating the data dividing out the other model, `flux_from`
	function interp_helper!(flux_to::AbstractMatrix, vars_to::AbstractMatrix, log_λ_to::AbstractVector,
		flux_from::AbstractMatrix,
		log_λ_data::AbstractMatrix; mask_extrema::Bool=true, keep_data_mask::Bool=true)
		try
			_spectra_interp_gp!(flux_to, vars_to, log_λ_to, d.flux ./ flux_from, d.var ./ (flux_from .^ 2), log_λ_data; gp_mean=1., keep_mask=keep_data_mask, λ_kernel=1/log_λ_gp_star)
		catch err
			if isa(err, DomainError)
				println("was unable to interpolate using a GP from one frame to another, using linear interpolation instead")
				y = d.flux ./ flux_from
				v = d.var ./ (flux_from .^ 2)
				for i in axes(y,2)
					interpolator1 = LinearInterpolation(view(y, :, i), view(log_λ_data, :, i))
					flux_to[:, i] = interpolator1.(log_λ_to)
					interpolator2 = LinearInterpolation(view(v, :, i), view(log_λ_data, :, i))
					vars_to[:, i] = interpolator2.(log_λ_to)
				end
			else
				rethrow()
			end
		end
		if mask_extrema
			mask_low_pixels!(flux_to, vars_to)
			mask_high_pixels!(flux_to, vars_to)
		end
	end
	interp_to_star!(om::OrderModel; kwargs...) = interp_helper!(flux_star, vars_star, om.star.log_λ,
		tel_model(om),
		d.log_λ_star; kwargs...)
	interp_to_tel!(om::OrderModel; kwargs...) = interp_helper!(flux_tel, vars_tel, om.tel.log_λ,
		star_model(om),
		d.log_λ_obs; kwargs...)
	# # a version of interp helper instead gets `flux_to` in the desired frame after dividing out a GP interpolated version of the other model
	# function interp_helper!(flux_to::AbstractMatrix, vars_to::AbstractMatrix, log_λ_to::AbstractVector,
	# 	flux_from::AbstractMatrix, vars_from::AbstractMatrix, log_λ_from::AbstractMatrix,
	# 	log_λ_data::AbstractMatrix; mask_extrema::Bool=true, keep_data_mask::Bool=true)
	# 	vars_from .= SOAP_gp_var
	# 	_spectra_interp_gp_div_gp!(flux_to, vars_to, log_λ_to, d.flux, d.var, log_λ_data, flux_from, vars_from, log_λ_from; keep_mask=keep_data_mask, ignore_model_uncertainty=true)
	# 	if mask_extrema
	# 		mask_low_pixels!(flux_to, vars_to)
	# 		mask_high_pixels!(flux_to, vars_to)
	# 	end
	# end
	# interp_to_star!(; kwargs...) = interp_helper!(flux_star, vars_star, om.star.log_λ,
	# 	flux_tel, vars_tel, tel_log_λ_star,
	# 	d.log_λ_star; kwargs...)
	# interp_to_tel!(; kwargs...) = interp_helper!(flux_tel, vars_tel, om.tel.log_λ,
	# 	flux_star, vars_star, star_log_λ_tel,
	# 	d.log_λ_obs; kwargs...)
	
	# if one wants to search for a telluric template
	if search_new_tel

		oms[2,1] = downsize(om, 0, 0)
		oms[2,1].star.lm.μ .= 1
		use_tel = χ²_star .> χ²_tel  # which pixels are telluric dominated

		# # modify use_tel to be more continuous
		# use_tel_window = 11
		# use_tel_density = 0.9
		# @assert isodd(use_tel_window)
		# _use_tel = χ²_star .> χ²_tel  # which pixels are telluric dominated
		# i = findfirst(_use_tel)
		# w = floor(Int, use_tel_window/2)
		# thres = floor(Int, use_tel_density * use_tel_window)
		# if !isnothing(i)
		# 	i += w+1
		# 	j = sum(view(_use_tel, (i-w-1):min(i+w-1, length(_use_tel))))  # look at first 11 use_tel
		# 	use_tel = zeros(Bool, length(_use_tel))
		# 	if j > thres; use_tel[(i-w-1):min(i+w-1, length(_use_tel))] .= true end
		# 	while (i+w+1) <= length(_use_tel)
		# 		j += _use_tel[i+w] - _use_tel[i-w-1]
		# 		if j > thres; use_tel[(i-w):min(i+w, length(_use_tel))] .= true end
		# 		i += 1
		# 	end
		# end


		if sum(use_tel) > 0  # if any pixels are telluric dominated

			# mask out the telluric dominated pixels for partial stellar template estimation
			_var = copy(d.var)
			_var[use_tel, :] .= Inf

			# get stellar template in portions of spectra where it is dominant
			_spectra_interp_gp!(flux_star, vars_star, oms[2,1].star.log_λ, d.flux, _var .+ SOAP_gp_var, d.log_λ_star; gp_mean=1., λ_kernel=1/log_λ_gp_star)
			oms[2,1].star.lm.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=use_mean)

			# get telluric template after dividing out the partial stellar template
			# flux_star .= oms[2,1].star.lm.μ
			interp_to_tel!(oms[2,1]; mask_extrema=false)
			oms[2,1].tel.lm.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)

			# get stellar template after dividing out full telluric template
			# flux_tel .= oms[2,1].tel.lm.μ
			interp_to_star!(oms[2,1]; mask_extrema=false)
			oms[2,1].star.lm.μ[:] = make_template(flux_star, vars_star; min=μ_min, max=μ_max, use_mean=use_mean)

		else

			# get telluric template after diving out full stellar template we already found
			fill_OrderModel!(oms[2,1], oms[1,1], 0:0, 0:0)
			# flux_star .= oms[2,1].star.lm.μ
			interp_to_tel!(oms[2,1]; mask_extrema=false)
			oms[2,1].tel.lm.μ[:] = make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)

		end

		if remove_reciprocal_continuum
			remove_reciprocal_continuum!(oms[2,1], flux_star, vars_star, flux_tel, vars_tel)
		end

		# optimize both templates
		mws = TotalWorkspace(oms[2,1], d)
		# flux_tel .= oms[2,1].tel.lm.μ
		# interp_to_star!(; mask_extrema=false)
		# mws.om.rv .= vec(project_doppler_comp(flux_star .- mws.om.star.lm.μ, doppler_component(mws.om.star.λ, mws.om.star.lm.μ), 1 ./ vars_star))
		om_add_tel = get_metrics!(mws, 2, 1)

	else
	
		om_add_tel = om_cur
	
	end

	j = comp2ind(n_tel_cur, n_star_cur)
	# if we looked for a telluric template, get the aic
	search_new_tel ? aic_tel = aics[comp2ind(n_tel_cur+1, n_star_cur)...] : aic_tel = Inf
	# was the model with a telluric template better than the current model (just a stellar template)?
	added_tel_better = aic_tel < aics[j...]
	
	# if including the telluric template model helped, use it going forward
	if added_tel_better; oms[j...] = om_cur end
	n_star_next = n_star_cur
	n_tel_next = n_tel_cur+added_tel_better
	added_tel_better ? aic_next = aic_tel : aic_next = aics[j...]
	
	add_comp = true
	println("looking for time variability...")

	# while we are looking to add new model components
	while add_comp

		if added_tel_better
			om_cur = om_add_tel
		else
			om_cur = om_add_star
		end

		# if we added a component, print some stuff
		if (n_tel_cur != n_tel_next) || (n_star_cur != n_star_next)
			println("n_comp: ($n_tel_cur,$n_star_cur) -> ($n_tel_next,$n_star_next)")
			println("aic   : $(aics[comp2ind(n_tel_cur, n_star_cur)...]) -> $(aics[comp2ind(n_tel_next, n_star_next)...])")
			println("RV std: $(rv_stds[comp2ind(n_tel_cur, n_star_cur)...]) -> $(rv_stds[comp2ind(n_tel_next, n_star_next)...])")
		end
		n_tel_cur, n_star_cur = n_tel_next, n_star_next
		search_new_tel = n_tel_cur+1 <= max_n_tel
		search_new_star = n_star_cur+1 <= max_n_star
		j = comp2ind(n_tel_cur, n_star_cur)

		# if we want to add a telluric component
		if search_new_tel

			i = comp2ind(n_tel_cur+1, n_star_cur)
			oms[i...] = downsize(om, n_tel_cur+1, n_star_cur)
			fill_OrderModel!(oms[i...], oms[j...], 1:n_tel_cur, 1:n_star_cur)
			oms[i...].rv .= 0  # the rv is a small effect that we could just be getting wrong
			# flux_star .= _eval_lm(oms[i...].star.lm)
			interp_to_tel!(oms[i...])# .+ rv_to_D(oms[i...].rv)')  # the rv is a small effect that we could just be getting wrong
			if n_tel_cur + 1 > 0  # if we are trying to add a feature vector
				EMPCA.EMPCA!(oms[i...].tel.lm, flux_tel, 1 ./ vars_tel; inds=(n_tel_cur+1):(n_tel_cur+1))
			else # if we are trying to add a template
				oms[i...].tel.lm.μ .= make_template(flux_tel, vars_tel; min=μ_min, max=μ_max, use_mean=use_mean)
			end
			# remove_reciprocal_continuum!(oms[i...], flux_star, vars_star, flux_tel, vars_tel)
			mws = TotalWorkspace(oms[i...], d)
			om_add_tel = get_metrics!(mws, i...)

		end

		# if we want to add a stellar component
		if search_new_star

			i = comp2ind(n_tel_cur, n_star_cur+1)
			oms[i...] = downsize(om, max(0, n_tel_cur), n_star_cur+1)
			fill_OrderModel!(oms[i...], oms[j...], 1:n_tel_cur, 1:n_star_cur)
			dop_comp .= doppler_component(oms[i...].star.λ, oms[i...].star.lm.μ)
			if n_tel_cur < 0  # if we don't have a telluric model
				oms[i...].tel.lm.μ .= 1
				# oms[i...].rv .=
				DEMPCA!(oms[i...].star.lm, copy(flux_star_no_tel), 1 ./ vars_star_no_tel, dop_comp; save_doppler_in_M1=false, inds=(n_star_cur+1):(n_star_cur+1), extra_vec=dop_comp)
				# remove_reciprocal_continuum!(oms[i...], flux_star, vars_star, flux_tel, vars_tel)
				mws = FrozenTelWorkspace(oms[i...], d)
			else  # if we have a telluric model
				# flux_tel .= _eval_lm(oms[i...].tel.lm)
				interp_to_star!(oms[i...])
				# oms[i...].rv .=
				DEMPCA!(oms[i...].star.lm, flux_star, 1 ./ vars_star, dop_comp; save_doppler_in_M1=false, inds=(n_star_cur+1):(n_star_cur+1), extra_vec=dop_comp)
				# remove_reciprocal_continuum!(oms[i...], flux_star, vars_star, flux_tel, vars_tel)
				mws = TotalWorkspace(oms[i...], d)
			end
			om_add_star = get_metrics!(mws, i...)
		end

		# look at the (up to 2) new aics and choose where to go next
		oms[j...] = om_cur
		search_new_tel ? aic_tel = aics[comp2ind(n_tel_cur+1, n_star_cur)...] : aic_tel = Inf
		search_new_star ? aic_star = aics[comp2ind(n_tel_cur, n_star_cur+1)...] : aic_star = Inf
		# println("tel: $aic_tel, star: $aic_star")
		added_tel_better = aic_tel < aic_star
		added_tel_better ? aic_next = aic_tel : aic_next = aic_star
		n_tel_next = n_tel_cur+added_tel_better
		n_star_next = n_star_cur+1-added_tel_better
		add_comp = (isfinite(aic_tel) || isfinite(aic_star)) && (!stop_early || aic_next < aics[j...]) && (search_new_tel || search_new_star)
	
	end

	println("stopped at ($n_tel_cur,$n_star_cur)")
	# aics[isnan.(aics)] .= Inf
	best_aic = argmin(aics)
	println("($(test_n_comp_tel[best_aic[1]]),$(test_n_comp_star[best_aic[2]])) was the best at aic = $(aics[best_aic])")
	println("best possible aic (k=0, χ²=0) = $(logdet_Σ + n * _log2π)")

	if return_full_path
		return oms, ℓs, aics, bics, rv_stds, rv_stds_intra, comp2ind, n_tel_cur, n_star_cur
	else
		if use_all_comps
			return oms[comp2ind(n_tel_cur, n_star_cur)...]
		else
			return oms[best_aic]
		end
	end
end
