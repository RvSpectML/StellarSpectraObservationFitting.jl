# using LineSearches
using ParameterHandling
using Optim
using Nabla
import Base.println

abstract type ModelWorkspace end
abstract type AdamWorkspace<:ModelWorkspace end

_χ²_loss(model, data, variance) = ((model .- data) .^ 2) ./ variance
_χ²_loss(model, data::Data) = _χ²_loss(model, data.flux, data.var)
__loss_diagnostic(tel, star, rv, d::GenericData) =
	_χ²_loss(total_model(tel, star, rv), d)
__loss_diagnostic(tel, star, rv, d::LSFData) =
	_χ²_loss(d.lsf * total_model(tel, star, rv), d)
__loss_diagnostic(tel, star, d::GenericData) =
	_χ²_loss(total_model(tel, star), d)
__loss_diagnostic(tel, star, d::LSFData) =
	_χ²_loss(d.lsf * total_model(tel, star), d)
function _loss_diagnostic(o::Output, om::OrderModel, d::Data;
	tel=nothing, star=nothing, rv=nothing)
    !isnothing(tel) ? tel_o = spectra_interp(_eval_lm_vec(om, tel), om.t2o) : tel_o = o.tel
	if typeof(om) <: OrderModelDPCA
		!isnothing(star) ? star_o = spectra_interp(_eval_lm_vec(om, star), om.b2o) : star_o = o.star
		!isnothing(rv) ? rv_o = spectra_interp(_eval_lm(om.rv.lm.M, rv), om.b2o) : rv_o = o.rv
		return __loss_diagnostic(tel_o, star_o, rv_o, d)
	end
	if !isnothing(star)
		if !isnothing(rv)
			star_o = spectra_interp(_eval_lm_vec(om, star), rv .+ om.bary_rvs, om.b2o)
		else
			star_o = spectra_interp(_eval_lm_vec(om, star), om.rv .+ om.bary_rvs, om.b2o)
		end
	elseif !isnothing(rv)
		star_o = spectra_interp(om.star.lm(), rv .+ om.bary_rvs, om.b2o)
	else
		star_o = o.star
	end
	return __loss_diagnostic(tel_o, star_o, d)
end
_loss_diagnostic(mws::ModelWorkspace; kwargs...) = _loss_diagnostic(mws.o, mws.om, mws.d; kwargs...)

# χ² loss function
_loss(tel, star, rv, d::Data) = sum(__loss_diagnostic(tel, star, rv, d))
_loss(tel, star, d::Data) = sum(__loss_diagnostic(tel, star, d))
function _loss(o::Output, om::OrderModel, d::Data;
	tel=nothing, star=nothing, rv=nothing)
    !isnothing(tel) ? tel_o = spectra_interp(_eval_lm_vec(om, tel), om.t2o) : tel_o = o.tel
	if typeof(om) <: OrderModelDPCA
		!isnothing(star) ? star_o = spectra_interp(_eval_lm_vec(om, star), om.b2o) : star_o = o.star
	    !isnothing(rv) ? rv_o = spectra_interp(_eval_lm(om.rv.lm.M, rv), om.b2o) : rv_o = o.rv
	    return _loss(tel_o, star_o, rv_o, d)
	end
	if !isnothing(star)
		if !isnothing(rv)
			star_o = spectra_interp(_eval_lm_vec(om, star), rv .+ om.bary_rvs, om.b2o)
		else
			star_o = spectra_interp(_eval_lm_vec(om, star), om.rv .+ om.bary_rvs, om.b2o)
		end
	elseif !isnothing(rv)
		star_o = spectra_interp(om.star.lm(), rv .+ om.bary_rvs, om.b2o)
	else
		star_o = o.star
	end
	return _loss(tel_o, star_o, d)
end
_loss(mws::ModelWorkspace; kwargs...) = _loss(mws.o, mws.om, mws.d; kwargs...)
function _loss_recalc_rv_basis(o::Output, om::OrderModel, d::Data; kwargs...)
	om.rv.lm.M .= calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.lm.μ)
	return _loss(o, om, d; kwargs...)
end
_loss_recalc_rv_basis(mws::ModelWorkspace; kwargs...) = _loss_recalc_rv_basis(mws.o, mws.om, mws.d; kwargs...)

function loss_func(mws::ModelWorkspace; include_priors::Bool=false)
	if include_priors
		return (; kwargs...) -> _loss(mws; kwargs...) + tel_prior(mws.om) + star_prior(mws.om)
	else
		return (; kwargs...) -> _loss(mws; kwargs...)
	end
end
function loss_funcs_telstar(o::Output, om::OrderModel, d::Data)
    l_telstar(telstar; kwargs...) =
        _loss(o, om, d; tel=telstar[1], star=telstar[2], kwargs...) +
			tel_prior(telstar[1], om) + star_prior(telstar[2], om)
	is_star_time_variable = is_time_variable(om.star)
    function l_telstar_s(telstar_s)
		if is_time_variable(om.tel)
			tel = [om.tel.lm.M, telstar_s[1], om.tel.lm.μ]
			is_star_time_variable ?
				star = [om.star.lm.M, telstar_s[2], om.star.lm.μ] : star = nothing
		elseif is_star_time_variable
			tel = nothing
			star = [om.star.lm.M, telstar_s[1], om.star.lm.μ]
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star)
    end

    l_rv(rv) = _loss(o, om, d; rv=rv)

    return l_telstar, l_telstar_s, l_rv
end
loss_funcs_telstar(mws::ModelWorkspace) = loss_funcs_telstar(mws.o, mws.om, mws.d)
function loss_funcs_total(o::Output, om::OrderModelDPCA, d::Data)
    l_total(total) =
		_loss_recalc_rv_basis(o, om, d; tel=total[1], star=total[2], rv=total[3]) +
		tel_prior(total[1], om) + star_prior(total[2], om)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
    function l_total_s(total_s)
		if is_tel_time_variable
			tel = [om.tel.lm.M, total_s[1], om.tel.lm.μ]
			is_star_time_variable ?
				star = [om.star.lm.M, total_s[2], om.star.lm.μ] : star = nothing
		elseif is_star_time_variable
			tel = nothing
			star = [om.star.lm.M, total_s[1], om.star.lm.μ]
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star, rv=total_s[1+is_star_time_variable+is_tel_time_variable])
    end

    return l_total, l_total_s
end
function loss_funcs_total(o::Output, om::OrderModelWobble, d::Data)
	l_total(total) =
		_loss(o, om, d; tel=total[1], star=total[2], rv=total[3]) +
		tel_prior(total[1], om) + star_prior(total[2], om)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
    function l_total_s(total_s)
		if is_tel_time_variable
			tel = [om.tel.lm.M, total_s[1], om.tel.lm.μ]
			is_star_time_variable ?
				star = [om.star.lm.M, total_s[2], om.star.lm.μ] : star = nothing
		elseif is_star_time_variable
			tel = nothing
			star = [om.star.lm.M, total_s[1], om.star.lm.μ]
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star, rv=total_s[1+is_star_time_variable+is_tel_time_variable])
    end

    return l_total, l_total_s
end
loss_funcs_total(mws::ModelWorkspace) = loss_funcs_total(mws.o, mws.om, mws.d)
function loss_funcs_frozen_tel(o::Output, om::OrderModel, d::Data)
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	function l_frozen_tel(total)
		is_tel_time_variable ? tel = [om.tel.lm.M, total[1], om.tel.lm.μ] : tel = nothing
		star = total[1+is_tel_time_variable]
		rv = total[2+is_tel_time_variable]
		return _loss(o, om, d; tel=tel, star=star, rv=rv) + star_prior(total[1+is_tel_time_variable], om)
	end
    function l_frozen_tel_s(total_s)
		if is_tel_time_variable
			tel = [om.tel.lm.M, total_s[1], om.tel.lm.μ]
			is_star_time_variable ?
				star = [om.star.lm.M, total_s[2], om.star.lm.μ] : star = nothing
		elseif is_star_time_variable
			tel = nothing
			star = [om.star.lm.M, total_s[1], om.star.lm.μ]
		else
			tel = nothing
			star = nothing
		end
		return _loss(o, om, d; tel=tel, star=star, rv=total_s[1+is_star_time_variable+is_tel_time_variable])
    end
    return l_frozen_tel, l_frozen_tel_s
end
loss_funcs_frozen_tel(mws::ModelWorkspace) = loss_funcs_frozen_tel(mws.o, mws.om, mws.d)

## Adam Version

α, β1, β2, ϵ = 2e-3, 0.9, 0.999, 1e-8
mutable struct Adam{T<:AbstractArray}
    α::Float64
    β1::Float64
    β2::Float64
    m::T
    v::T
    β1_acc::Float64
    β2_acc::Float64
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
Base.copy(opt::Adam) = Adam(opt.α, opt.β1, opt.β2, opt.m, opt.v, opt.β1_acc, opt.β2_acc, opt.ϵ)

mutable struct AdamState
    iter::Int
    ℓ::Float64
    L1_Δ::Float64
    L2_Δ::Float64
    L∞_Δ::Float64
	δ_ℓ::Float64
	δ_L1_Δ::Float64
	δ_L2_Δ::Float64
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
function iterate!(θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam)
	α=opt.α; β1=opt.β1; β2=opt.β2; ϵ=opt.ϵ; β1_acc=opt.β1_acc; β2_acc=opt.β2_acc; m=opt.m; v=opt.v
    # the matrix and dotted version is slower
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + (1.0 - β1) * ∇θ[n]
        v[n] = β2 * v[n] + (1.0 - β2) * ∇θ[n]^2
        m̂ = m[n] / (1 - β1_acc)
        v̂ = v[n] / (1 - β2_acc)
        θ[n] -= α * m̂ / (sqrt(v̂) + ϵ)
    end
	β1_acc *= β1
	β2_acc *= β2
end

function AdamState!_helper(as::AdamState, f::Symbol, val)
	setfield!(as, Symbol(:δ_,f), val / getfield(as, f))
	setfield!(as, f, val)
end
function AdamState!(as::AdamState, ℓ, Δ)
	as.iter += 1
	AdamState!_helper(as, :ℓ, ℓ)
	flat_Δ = Iterators.flatten(Iterators.flatten(Δ))
	AdamState!_helper(as, :L1_Δ, L1(flat_Δ))
	AdamState!_helper(as, :L2_Δ, L2(flat_Δ))
	AdamState!_helper(as, :L∞_Δ, L∞(Δ))
end

_print_stuff_def = false
_iter_def = 100
_f_reltol_def = 1e-4
_g_reltol_def = 1e-3
_g_L∞tol_def = 1e3

struct AdamSubWorkspace{T}
	θ::T
	opt
	as::AdamState
	l::Function
	gl::Function
	function AdamSubWorkspace(θ::T, opt, as, l, gl) where T
		@assert typeof(l(θ)) <: Real
		return new{T}(θ, opt, as, l, gl)
	end
end
function AdamSubWorkspace(θ, l::Function)
	gl = ∇(l; get_output=true)
	gl(θ)  # compile it
	return AdamSubWorkspace(θ, Adams(θ), AdamState(), l, gl)
end

function update!(aws::AdamSubWorkspace)
    val, Δ = aws.gl(aws.θ)
	Δ = only(Δ)
	AdamState!(aws.as, val.val, Δ)
    iterate!(aws.θ, Δ, aws.opt)
end

function check_converged(as::AdamState, f_reltol::Real, g_reltol::Real, g_L∞tol::Real)
	as.ℓ > 0 ? δ_ℓ = as.δ_ℓ : δ_ℓ = 1 / as.δ_ℓ  # further reductions in negative cost functions are good!
	return ((δ_ℓ > (1 - f_reltol)) && (max(as.δ_L2_Δ,1/as.δ_L2_Δ) < (1+abs(g_reltol)))) || (as.L∞_Δ < g_L∞tol)
end
check_converged(as::AdamState, iter::Int, f_reltol::Real, g_reltol::Real, g_L∞tol::Real) = (as.iter > iter) || check_converged(as, f_reltol, g_reltol, g_L∞tol)
function train_SubModel!(aws::AdamSubWorkspace; iter=_iter_def, f_reltol = _f_reltol_def, g_reltol = _g_reltol_def, g_L∞tol = _g_L∞tol_def, cb::Function=(as::AdamState)->(), kwargs...)
	converged = false  # check_converged(aws.as, iter, f_tol, g_tol)
	while !converged
		update!(aws)
		cb(aws.as)
		converged = check_converged(aws.as, iter, f_reltol, g_reltol, g_L∞tol)
	end
	converged = check_converged(aws.as, f_reltol, g_reltol, g_L∞tol)
	# converged ? println("Converged") : println("Max Iters Reached")
	return converged
end

rel_step_size(θ::AbstractVecOrMat) = sqrt(mean(abs2, θ))
function scale_α_helper!(opt::Adam, α_ratio::Real, θ::AbstractVecOrMat, α::Real, scale_α::Bool)
	scale_α ? opt.α = α_ratio * rel_step_size(θ) : opt.α = α
end
function scale_α_helper!(opts::Vector, α_ratio::Real, θs, α::Real, scale_α::Bool)
	@inbounds for i in eachindex(opts)
		scale_α_helper!(opts[i], α_ratio, θs[i], α, scale_α)
	end
end
_scale_α_def = false

struct TotalWorkspace <: AdamWorkspace
	total::AdamSubWorkspace
	om::OrderModel
	o::Output
	d::Data
	only_s::Bool
end

function TotalWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def)
	l_total, l_total_s = loss_funcs_total(o, om, d)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	typeof(om) <: OrderModelDPCA ? rvs = om.rv.lm.s : rvs = om.rv
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				total = AdamSubWorkspace([om.tel.lm.s, om.star.lm.s, rvs], l_total_s)
			else
				total = AdamSubWorkspace([om.tel.lm.s, rvs], l_total_s)
			end
		elseif is_star_time_variable
			total = AdamSubWorkspace([om.star.lm.s, rvs], l_total_s)
		else
			total = AdamSubWorkspace([rvs], l_total_s)
		end
	else
		total = AdamSubWorkspace([vec(om.tel.lm), vec(om.star.lm), rvs], l_total)
	end
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


struct FrozenTelWorkspace <: AdamWorkspace
	total::AdamSubWorkspace
	om::OrderModel
	o::Output
	d::Data
	only_s::Bool
end

function FrozenTelWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def)
	l_frozen_tel, l_frozen_tel_s = loss_funcs_frozen_tel(o, om, d)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	is_tel_time_variable = is_time_variable(om.tel)
	is_star_time_variable = is_time_variable(om.star)
	typeof(om) <: OrderModelDPCA ? rvs = om.rv.lm.s : rvs = om.rv
	if only_s
		if is_tel_time_variable
			if is_star_time_variable
				total = AdamSubWorkspace([om.tel.lm.s, om.star.lm.s, rvs], l_frozen_tel_s)
			else
				total = AdamSubWorkspace([om.tel.lm.s, rvs], l_frozen_tel_s)
			end
		elseif is_star_time_variable
			total = AdamSubWorkspace([om.star.lm.s, rvs], l_frozen_tel_s)
		else
			total = AdamSubWorkspace([rvs], l_frozen_tel_s)
		end
	else
		is_tel_time_variable ?
			total = AdamSubWorkspace([om.tel.lm.s, vec(om.star.lm), rvs], l_frozen_tel) :
			total = AdamSubWorkspace([vec(om.star.lm), rvs], l_frozen_tel)
	end
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

function train_OrderModel!(mws::AdamWorkspace; ignore_regularization::Bool=false, print_stuff::Bool=_print_stuff_def, shift_scores::Bool=true, kwargs...)

    if ignore_regularization
        reg_tel_holder = copy(mws.om.reg_tel)
        reg_star_holder = copy(mws.om.reg_star)
        rm_regularization!(mws.om)
    end

	function cb(as::AdamState)
		if shift_scores
			if !(typeof(mws) <: FrozenTelWorkspace)
				remove_lm_score_means!(mws.om.tel.lm)
			end
			if typeof(mws.om) <: OrderModelWobble
				remove_lm_score_means!(mws.om.star.lm)
			end
		end
		if print_stuff; println(as) end
	end
	# print_stuff ? cb(as::AdamState) = println(as) : cb(as::AdamState) = nothing

	result = train_SubModel!(mws.total; cb=cb, kwargs...)
	mws.total.as.iter = 0
    Output!(mws)

    if ignore_regularization
        copy_dict!(mws.om.reg_tel, reg_tel_holder)
        copy_dict!(mws.om.reg_star, reg_star_holder)
    end
	return result
end

Output!(mws::ModelWorkspace) = Output!(mws.o, mws.om, mws.d)


## Optim Versions

function opt_funcs(loss::Function, pars::AbstractVecOrMat)
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns Vector of untransformed params
    f = loss ∘ unflatten
	g_nabla = ∇(loss)
	g_val_nabla = ∇(loss; get_output=true)
	g_nabla(pars)  # compile it
	g_val_nabla(pars)  # compile it
    function g!(G, θ)
        G[:], _ = flatten(g_nabla(unflatten(θ)))
    end
    function fg_obj!(G, θ)
		l, g = g_val_nabla(unflatten(θ))
		G[:], _ = flatten(g)
        return l.val
    end
    return flat_initial_params, OnceDifferentiable(f, g!, fg_obj!, flat_initial_params), unflatten
end

struct OptimSubWorkspace
    θ::AbstractVecOrMat
    obj::OnceDifferentiable
    opt::Optim.FirstOrderOptimizer
    p0::Vector
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

struct OptimWorkspace <: ModelWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
end
function OptimWorkspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
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
	return OptimWorkspace(telstar, rv, om, o, d, only_s)
end
OptimWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	OptimWorkspace(om(inds), d(inds); kwargs...)
OptimWorkspace(om::OrderModel, d::Data; kwargs...) =
	OptimWorkspace(om, Output(om, d), d; kwargs...)

function _OSW_optimize!(osw::OptimSubWorkspace, options::Optim.Options)
    result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)
    osw.p0[:] = result.minimizer
    return result
end

function optim_print(x::OptimizationState)
	println()
	if x.iteration > 0
		println("Iter:  ", x.iteration)
		println("Time:  ", x.metadata["time"], " s")
		println("ℓ:     ", x.value)
		println("l∞(∇): ", x.g_norm)
		println()
	end
	return false
end
# ends optimization if true
function optim_cb_f(; print_stuff::Bool=true)
    if print_stuff
		return (x::OptimizationState) -> optim_print(x::OptimizationState)
    else
		return (x::OptimizationState) -> false
    end
end

function train_OrderModel!(ow::OptimWorkspace; print_stuff::Bool=_print_stuff_def, iter::Int=_iter_def, f_tol::Real=_f_reltol_def, g_tol::Real=_g_L∞tol_def, train_telstar::Bool=true, ignore_regularization::Bool=false, kwargs...)
    optim_cb = optim_cb_f(; print_stuff=print_stuff)

    if ignore_regularization
        reg_tel_holder = copy(ow.om.reg_tel)
        reg_star_holder = copy(ow.om.reg_star)
        rm_regularization!(ow.om)
    end

    if train_telstar
        options = Optim.Options(;iterations=iter, f_tol=f_tol, g_tol=g_tol, callback=optim_cb, kwargs...)
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
	result_rv = train_rvs_optim!(ow, optim_cb, kwargs...)
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
function train_rvs_optim!(rv_ws::OptimSubWorkspace, rv::Submodel, star::Submodel, optim_cb::Function, kwargs...)
	options = Optim.Options(; callback=optim_cb, g_tol=1e-2, kwargs...)
	rv.lm.M .= calc_doppler_component_RVSKL(star.λ, star.lm.μ)
	result_rv = _OSW_optimize!(rv_ws, options)
	rv.lm.s[:] = rv_ws.unflatten(rv_ws.p0)
	return result_rv
end
function train_rvs_optim!(rv_ws::OptimSubWorkspace, rv::AbstractVector, optim_cb::Function, kwargs...)
	options = Optim.Options(; callback=optim_cb, g_tol=1e-2, kwargs...)
	result_rv = _OSW_optimize!(rv_ws, options)
	rv[:] = rv_ws.unflatten(rv_ws.p0)
	return result_rv
end
train_rvs_optim!(ow::OptimWorkspace, optim_cb::Function, kwargs...) =
	typeof(ow.om) <: OrderModelDPCA ?
		train_rvs_optim!(ow.rv, ow.om.rv, ow.om.star, optim_cb, kwargs...) :
		train_rvs_optim!(ow.rv, ow.om.rv, optim_cb, kwargs...)

function finalize_scores_setup(mws::ModelWorkspace; print_stuff::Bool=_print_stuff_def, kwargs...)
	if is_time_variable(mws.om.tel) || is_time_variable(mws.om.star)
		mws_s = OptimWorkspace(mws.om, mws.d; only_s=true)
		score_trainer() = train_OrderModel!(mws_s; kwargs...)
		return score_trainer
	end
	optim_cb=optim_cb_f(; print_stuff=print_stuff)
	loss_rv(rv) = _loss(mws; rv=rv)
	return _finalize_scores_setup(mws, mws.om, loss_rv, optim_cb; kwargs...)
end
function _finalize_scores_setup(mws::ModelWorkspace, om::OrderModelDPCA, loss_rv::Function, optim_cb::Function; kwargs...)
	rv_ws = OptimSubWorkspace(mws.om.rv.lm.s, loss_rv; use_cg=true)
	rv_trainer() = train_rvs_optim!(rv_ws, mws.om.rv, mws.om.star, optim_cb, kwargs...)
	return rv_trainer
end
function _finalize_scores_setup(mws::ModelWorkspace, om::OrderModelWobble, loss_rv::Function, optim_cb::Function; kwargs...)
	rv_ws = OptimSubWorkspace(mws.om.rv, loss_rv; use_cg=true)
	rv_trainer() = train_rvs_optim!(rv_ws, mws.om.rv, optim_cb, kwargs...)
	return rv_trainer
end
function finalize_scores!(score_trainer::Function, mws::ModelWorkspace)
	score_trainer()
	Output!(mws)
end
function finalize_scores!(mws::ModelWorkspace; kwargs...)
	score_trainer = finalize_scores_setup(mws, kwargs...)
	finalize_scores!(score_trainer, mws)
end

is_time_variable(lm::LinearModel) = !(typeof(lm) <: TemplateModel)
is_time_variable(sm::Submodel) = is_time_variable(sm.lm)
