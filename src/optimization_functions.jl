# using LineSearches
using ParameterHandling
using Optim
using Nabla
import Base.println

abstract type ModelWorkspace end

# χ² loss function
_loss(tel, star, rv, d::GenericData) =
    sum(((total_model(tel, star, rv) .- d.flux) .^ 2) ./ d.var)
# χ² loss function broadened by an lsf at each time (about 2x slower)
# _loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::LSFData) =
#     mapreduce(i -> sum((((d.lsf[i] * (view(tel, :, i) .* (view(star, :, i) .+ view(rv, :, i)))) .- view(d.flux, :, i)) .^ 2) ./ view(d.var, :, i)), +, 1:size(tel, 2))
_loss(tel, star, rv, d::LSFData) =
    sum((((d.lsf * total_model(tel, star, rv)) .- d.flux) .^ 2) ./ d.var)
function _loss(o::Output, om::OrderModel, d::Data;
	tel=nothing, star=nothing, rv=nothing)
    !isnothing(tel) ? tel_o = spectra_interp(_eval_lm_vec(om, tel), om.t2o) : tel_o = o.tel
    !isnothing(star) ? star_o = spectra_interp(_eval_lm_vec(om, star), om.b2o) : star_o = o.star
    !isnothing(rv) ? rv_o = spectra_interp(_eval_lm_vec(om, rv), om.b2o) : rv_o = o.rv
    return _loss(tel_o, star_o, rv_o, d)
end
_loss(mws::ModelWorkspace; kwargs...) = _loss(mws.o, mws.om, mws.d; kwargs...)
function _loss_recalc_rv_basis(om::OrderModel, d::Data, tel, star, rv_s)
	om.rv.lm.M .= calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.lm.μ)
    return _loss(spectra_interp(_eval_lm_vec(om, tel), om.t2o),
	        spectra_interp(_eval_lm_vec(om, star), om.b2o),
			spectra_interp(_eval_lm(om.rv.lm.M, rv_s), om.b2o), d)
end
_loss_recalc_rv_basis(mws::ModelWorkspace, tel, star, rv_s) = _loss_recalc_rv_basis(mws.om, mws.d, tel, star, rv_s)

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
			model_prior(telstar[1], om.reg_tel) + model_prior(telstar[2], om.reg_star)
    function l_telstar_s(telstar_s)
		!(typeof(om.tel.lm) <: TemplateModel) ? tel = [om.tel.lm.M, telstar_s[1], om.tel.lm.μ] : tel = nothing
		!(typeof(om.tel.lm) <: TemplateModel) ? star = [om.star.lm.M, telstar_s[2], om.star.lm.μ] : star = nothing
		return _loss(o, om, d; tel=tel, star=star) +
			model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv_s) = _loss(o, om, d; rv=[om.rv.lm.M, rv_s])

    return l_telstar, l_telstar_s, l_rv
end
loss_funcs_telstar(mws::ModelWorkspace) = loss_funcs_telstar(mws.o, mws.om, mws.d)
function loss_funcs_total(o::Output, om::OrderModel, d::Data)
    l_total(total) =
		_loss_recalc_rv_basis(om, d, total[1], total[2], total[3]) +
		model_prior(total[1], om.reg_tel) + model_prior(total[2], om.reg_star)
    function l_total_s(total_s)
		!(typeof(om.tel.lm) <: TemplateModel) ? tel = [om.tel.lm.M, total_s[1], om.tel.lm.μ] : tel = nothing
		!(typeof(om.star.lm) <: TemplateModel) ? star = [om.star.lm.M, total_s[2], om.star.lm.μ] : star = nothing
		rv = [om.rv.lm.M, total_s[3]]
		return _loss(o, om, d; tel=tel, star=star, rv=rv) +
			model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    return l_total, l_total_s
end
loss_funcs_total(mws::ModelWorkspace) = loss_funcs_total(mws.o, mws.om, mws.d)

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

# L∞_cust(Δ) = maximum([maximum([maximum(i) for i in j]) for j in Δ])
function AdamState!_helper(as::AdamState, f::Symbol, val)
	setfield!(as, Symbol(:δ_,f), val / getfield(as, f))
	setfield!(as, f, val)
end
function AdamState!(as::AdamState, ℓ, Δ)
	as.iter += 1
	AdamState!_helper(as, :ℓ, ℓ)
	flat_Δ = Iterators.flatten(Iterators.flatten(Δ))
	AdamState!_helper(as, :L1_Δ, sum(abs, flat_Δ))
	AdamState!_helper(as, :L2_Δ, sum(abs2, flat_Δ))
	AdamState!_helper(as, :L∞_Δ, L∞(Δ))
end

_print_stuff_def = false
_iter_def = 100
_f_reltol_def = 1e-6
_g_reltol_def = 1e-3
_g_L∞tol_def = 1e3

struct AdamWorkspace{T}
	θ::T
	opt
	as::AdamState
	l::Function
	function AdamWorkspace(θ::T, opt, as, l) where T
		@assert typeof(l(θ)) <: Real
		return new{T}(θ, opt, as, l)
	end
end
AdamWorkspace(θ, l::Function) = AdamWorkspace(θ, Adams(θ), AdamState(), l)

function update!(aws::AdamWorkspace)
    val, Δ = ∇(aws.l; get_output=true)(aws.θ)
	Δ = only(Δ)
	AdamState!(aws.as, val.val, Δ)
    iterate!(aws.θ, Δ, aws.opt)
end

check_converged(as::AdamState, f_reltol::Real, g_reltol::Real, g_L∞tol::Real) = ((max(as.δ_ℓ, 1/as.δ_ℓ) < (1 + abs(f_reltol))) && (max(as.δ_L2_Δ,1/as.δ_L2_Δ) < (1+abs(g_reltol)))) || (as.L∞_Δ < g_L∞tol)
check_converged(as::AdamState, iter::Int, f_reltol::Real, g_reltol::Real, g_L∞tol::Real) = (as.iter > iter) || check_converged(as, f_reltol, g_reltol, g_L∞tol)
function train_SubModel!(aws::AdamWorkspace; iter=_iter_def, f_reltol = _f_reltol_def, g_reltol = _g_reltol_def, g_L∞tol = _g_L∞tol_def, cb::Function=()->(), kwargs...)
	converged = false  # check_converged(aws.as, iter, f_tol, g_tol)
	while !converged
		update!(aws)
		cb()
		converged = check_converged(aws.as, iter, f_reltol, g_reltol, g_L∞tol)
	end
	converged = check_converged(aws.as, f_reltol, g_reltol, g_L∞tol)
	# converged ? println("Converged") : println("Max Iters Reached")
	return converged
end
function default_cb(as::AdamState; print_stuff=_print_stuff_def)
	if print_stuff
		return () -> println(as)
	else
		return () -> ()
	end
end

rel_step_size(θ::AbstractVecOrMat) = sqrt(mean(abs2, θ))
function scale_α_helper!(opt::Adam, α_ratio::Real, θ::AbstractVecOrMat, α::Real, scale_α::Bool)
	scale_α ? opt.α = α_ratio * rel_step_size(θ) : opt.α = α
end
function scale_α_helper!(opts::Vector, α_ratio::Real, θs, α::Real, scale_α::Bool)
	for i in eachindex(opts)
		scale_α_helper!(opts[i], α_ratio, θs[i], α, scale_α)
	end
end
_scale_α_def = false

struct TelStarWorkspace <: ModelWorkspace
	telstar::AdamWorkspace
	rv::AdamWorkspace
	om::OrderModel
	o::Output
	d::Data
	only_s::Bool
end
function TelStarWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def)
	loss_telstar, loss_telstar_s, loss_rv = loss_funcs_telstar(o, om, d)
	only_s ?
		telstar = AdamWorkspace([om.tel.lm.s, om.star.lm.s], loss_telstar_s) :
		telstar = AdamWorkspace([vec(om.tel.lm), vec(om.star.lm)], loss_telstar)
	rv = AdamWorkspace(om.rv.lm.s, loss_rv)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	scale_α_helper!(telstar.opt, α_ratio, telstar.θ, α, scale_α)
	scale_α_helper!(rv.opt, α_ratio, rv.θ, α, true)
	return TelStarWorkspace(telstar, rv, om, o, d, only_s)
end
TelStarWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	TelStarWorkspace(om(inds), d(inds); kwargs...)
TelStarWorkspace(om::OrderModel, d::Data; kwargs...) =
	TelStarWorkspace(Output(om, d), om, d; kwargs...)

_telstar_iters_per_loop = 50
function train_OrderModel!(mw::TelStarWorkspace; train_telstar::Bool=true, ignore_regularization::Bool=false, print_stuff::Bool=_print_stuff_def, iters_per_loop::Int=_telstar_iters_per_loop, iter=_iter_def, kwargs...)

    if ignore_regularization
        reg_tel_holder = copy(mw.om.reg_tel)
        reg_star_holder = copy(mw.om.reg_star)
        zero_regularization(mw.om)
    end
	n_loop = Int(ceil(iter//iters_per_loop))
	for i in 1:n_loop
		i == n_loop ? current_iter = iter % iters_per_loop : current_iter = iters_per_loop
	    if train_telstar
			cb_telstar = default_cb(mw.telstar.as; print_stuff)
			result_telstar = train_SubModel!(mw.telstar; cb=cb_telstar, iter=current_iter, kwargs...)
			mw.telstar.as.iter = 0
	        mw.o.star .= star_model(mw.om)
	        mw.o.tel .= tel_model(mw.om)
	    end
		mw.om.rv.lm.M .= calc_doppler_component_RVSKL(mw.om.star.λ, mw.om.star.lm.μ)
		cb_rv = default_cb(mw.rv.as; print_stuff)
		result_rv = train_SubModel!(mw.rv; cb=cb_rv, iter=current_iter, kwargs...)
	    mw.rv.as.iter = 0
		mw.o.rv .= rv_model(mw.om)
	end
	recalc_total!(mw.o, mw.d)

    if ignore_regularization
        copy_dict!(mw.om.reg_tel, reg_tel_holder)
        copy_dict!(mw.om.reg_star, reg_star_holder)
    end
	# return result_telstar, result_rv
end


struct TotalWorkspace <: ModelWorkspace
	total::AdamWorkspace
	om::OrderModel
	o::Output
	d::Data
	only_s::Bool
end

function TotalWorkspace(o::Output, om::OrderModel, d::Data; only_s::Bool=false, α::Real=α, scale_α::Bool=_scale_α_def)
	l_total, l_total_s = loss_funcs_total(o, om, d)
	only_s ?
		total = AdamWorkspace([om.tel.lm.s, om.star.lm.s, om.rv.lm.s], l_total_s) :
		total = AdamWorkspace([vec(om.tel.lm), vec(om.star.lm), om.rv.lm.s], l_total)
	α_ratio = α * sqrt(length(om.tel.lm.μ)) # = α / rel_step_size(om.tel.lm.M) assuming M starts as L2 normalized basis vectors. Need to use this instead because TemplateModels don't have basis vectors
	scale_α_helper!(total.opt[1:2], α_ratio, total.θ, α, scale_α)
	scale_α_helper!(total.opt[3], α_ratio, total.θ[3], α, true)
	return TotalWorkspace(total, om, o, d, only_s)
end
TotalWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	TotalWorkspace(om(inds), d(inds); kwargs...)
TotalWorkspace(om::OrderModel, d::Data; kwargs...) =
	TotalWorkspace(Output(om, d), om, d; kwargs...)

function train_OrderModel!(mws::TotalWorkspace; ignore_regularization::Bool=false, print_stuff::Bool=_print_stuff_def, kwargs...)

    if ignore_regularization
        reg_tel_holder = copy(mws.om.reg_tel)
        reg_star_holder = copy(mws.om.reg_star)
        zero_regularization(mws.om)
    end

	cb = default_cb(mws.total.as; print_stuff)
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

possible_θ = Union{Vector{<:Vector{<:AbstractArray}}, Vector{<:Array}, AbstractMatrix}

function opt_funcs(loss::Function, pars::possible_θ)
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns Vector of untransformed params
    f = loss ∘ unflatten
    function g!(G, θ)
		θunfl = unflatten(θ)
        G[:], _ = flatten(∇(loss)(θunfl))
    end
    function fg_obj!(G, θ)
		θunfl = unflatten(θ)
		l, g = ∇(loss; get_output=true)(θunfl)
		G[:], _= flatten(g)
        return l.val
    end
    return flat_initial_params, OnceDifferentiable(f, g!, fg_obj!, flat_initial_params), unflatten
end

struct OptimSubWorkspace
    θ::possible_θ
    obj::OnceDifferentiable
    opt::Optim.FirstOrderOptimizer
    p0::Vector
    unflatten::Function
end
function OptimSubWorkspace(θ::possible_θ, loss::Function; use_cg::Bool=true)
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
	rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true)
	only_s ?
		telstar = OptimSubWorkspace([om.tel.lm.s, om.star.lm.s], loss_telstar_s; use_cg=!only_s) :
		telstar = OptimSubWorkspace([vec(om.tel.lm), vec(om.star.lm)], loss_telstar; use_cg=!only_s)
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
        zero_regularization(ow.om)
    end

    if train_telstar
        options = Optim.Options(;iterations=iter, f_tol=f_tol, g_tol=g_tol, callback=optim_cb, kwargs...)
        # optimize tellurics and star
        result_telstar = _OSW_optimize!(ow.telstar, options)
		lm_vec = ow.telstar.unflatten(ow.telstar.p0)
        if ow.only_s
			ow.om.tel.lm.s[:] = lm_vec[1]
			ow.om.star.lm.s[:] = lm_vec[2]
        else
            copy_to_LinearModel!(ow.om.tel.lm, lm_vec[1])
			copy_to_LinearModel!(ow.om.star.lm, lm_vec[2])
        end
        ow.o.star .= star_model(ow.om)
        ow.o.tel .= tel_model(ow.om)
    end

    # optimize RVs
    options = Optim.Options(;callback=optim_cb, g_tol=1e-2, kwargs...)
    ow.om.rv.lm.M .= calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    result_rv = _OSW_optimize!(ow.rv, options)
	ow.om.rv.lm.s[:] = ow.rv.unflatten(ow.rv.p0)
    ow.o.rv .= rv_model(ow.om)
	recalc_total!(ow.o, ow.d)

    if ignore_regularization
        copy_dict!(ow.om.reg_tel, reg_tel_holder)
        copy_dict!(mw.om.reg_star, reg_star_holder)
    end
    return result_telstar, result_rv
end

fine_train_OrderModel!(mws::ModelWorkspace; iter=3*_iter_def, kwargs...) =
	train_OrderModel!(mws; iter=iter, kwargs...)
