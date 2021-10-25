# using LineSearches
using ParameterHandling
using Optim
using Nabla

# χ² loss function
_loss(tel, star, rv, d::GenericData) =
    sum(((total_model(tel, star, rv) .- d.flux) .^ 2) ./ d.var)
# χ² loss function broadened by an lsf at each time (about 2x slower)
# _loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::LSFData) =
#     mapreduce(i -> sum((((d.lsf[i] * (view(tel, :, i) .* (view(star, :, i) .+ view(rv, :, i)))) .- view(d.flux, :, i)) .^ 2) ./ view(d.var, :, i)), +, 1:size(tel, 2))
_loss(tel, star, rv, d::LSFData) =
    sum((((d.lsf * total_model(tel, star, rv)) .- d.flux) .^ 2) ./ d.var)
function loss(o::Output, om::OrderModel, d::Data;
	tel=om.tel.lm, star=om.star.lm, rv=om.rv.lm,
	recalc_tel::Bool=true, recalc_star::Bool=true, recalc_rv::Bool=true)

    recalc_tel ? tel_o = spectra_interp(_eval_lm(tel), om.t2o) : tel_o = o.tel
    recalc_star ? star_o = spectra_interp(_eval_lm(star), om.b2o) : star_o = o.star
    recalc_rv ? rv_o = spectra_interp(_eval_lm(rv), om.b2o) : rv_o = o.rv
    return _loss(tel_o, star_o, rv_o, d)
end

possible_θ = Union{Vector{<:Vector{<:AbstractArray}}, AbstractMatrix}

function loss_funcs_telstar(o::Output, om::OrderModel, d::Data)
    l(; kwargs...) = loss(o, om, d; kwargs...)

    l_telstar(telstar; kwargs...) =
        loss(o, om, d; tel=telstar[1], star=telstar[2], recalc_rv=false, kwargs...) +
			model_prior(telstar[1], om.reg_tel) + model_prior(telstar[2], om.reg_star)
    function l_telstar_s(telstar_s)
		recalc_tel = !(typeof(om.tel.lm) <: TemplateModel)
		recalc_star = !(typeof(om.tel.lm) <: TemplateModel)
		recalc_tel ? tel = [om.tel.lm.M, telstar_s[1], om.tel.lm.μ] : tel = om.tel.lm
		recalc_star ? star = [om.star.lm.M, telstar_s[2], om.star.lm.μ] : star = om.star.lm
		return loss(o, om, d; tel=tel, star=star, recalc_rv=false, recalc_tel=recalc_tel, recalc_star=recalc_star) +
			model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv_s) = loss(o, om, d; rv=[om.rv.lm.M, rv_s], recalc_tel=false, recalc_star=false)

    return l, l_telstar, l_telstar_s, l_rv
end


function opt_funcs(loss::Function, pars::possible_θ)
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns Vector of untransformed params
    f = loss ∘ unflatten
    function g!(G, θ)
		θunfl = unflatten(θ)
        G[:], _= flatten(∇(loss)(θunfl))
    end
    function fg_obj!(G, θ)
		θunfl = unflatten(θ)
		l, g = ∇(loss; get_output=true)(θunfl)
		G[:], _= flatten(g)
        return l
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
	use_cg ? opt = ConjugateGradient() : opt = LBFGS()
	# initial_state(method::LBFGS, ...) doesn't use the options for anything
	return OptimSubWorkspace(θ, obj, opt, p0, unflatten)
end

struct OptimWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
end
function OptimWorkspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
	loss, loss_telstar, loss_telstar_s, loss_rv = loss_funcs_telstar(o, om, d)
	rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true)
	only_s ?
		telstar = OptimSubWorkspace([om.tel.lm.s, om.star.lm.s], loss_telstar_s; use_cg=!only_s) :
		telstar = OptimSubWorkspace([vec(om.tel.lm), vec(om.star.lm)], loss_telstar; use_cg=!only_s)
	ow = OptimWorkspace(telstar, rv, om, o, d, only_s)
	if return_loss_f
		return ow, loss
	else
		return ow
	end
end
OptimWorkspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
	OptimWorkspace(om(inds), d(inds); kwargs...)
OptimWorkspace(om::OrderModel, d::Data; kwargs...) =
	OptimWorkspace(om, Output(om, d), d; kwargs...)

function _OSW_optimize!(osw::OptimSubWorkspace, options::Optim.Options; return_result::Bool=true)
    result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)
    osw.p0 .= result.minimizer
    if return_result
        return result, osw.unflatten(osw.p0)
    else
        return osw.unflatten(osw.p0)
    end
end

# ends optimization if true
function optim_cb(x::OptimizationState; print_stuff::Bool=true)
    if print_stuff
        println()
        if x.iteration > 0
            println("Iter:  ", x.iteration)
            println("Time:  ", x.metadata["time"], " s")
            println("ℓ:     ", x.value)
            println("l2(∇): ", x.g_norm)
            println()
        end
    end
    return false
end


_print_stuff_def = false
_iter_def = 100
_f_tol_def = 1e-6
_g_tol_def = 400

function _custom_copy!(from::NamedTuple, to...)
	for (i, k) in enumerate(keys(from))
        @assert typeof(from[k])==typeof(to[i])
        if typeof(to[i])<:LinearModel
            copy_LinearModel!(from[k], to[i])
        elseif typeof(to[i])<:AbstractVector
            to[i] .= from[k]
        else
            @error "didn't expect an object of type $(typeof(to[i]))"
        end
	end
end


function train_OrderModel!(ow::OptimWorkspace; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def*sqrt(length(ow.telstar.p0)), train_telstar::Bool=true, ignore_regularization::Bool=false, kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)

    if ignore_regularization
        reg_tel_holder = copy(ow.om.reg_tel)
        reg_star_holder = copy(ow.om.reg_star)
        zero_regularization(ow.om)
    end

    if train_telstar
        options = Optim.Options(;iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
        # optimize tellurics and star
        result_telstar, nt = _OSW_optimize!(ow.telstar, options)
        if ow.only_s
            _custom_copy!(nt, ow.om.tel.lm.s, ow.om.star.lm.s)
        else
            _custom_copy!(nt, ow.om.tel.lm, ow.om.star.lm)
        end
        ow.o.star .= star_model(ow.om)
        ow.o.tel .= tel_model(ow.om)
    end

    # optimize RVs
    options = Optim.Options(;callback=optim_cb_local, g_tol=g_tol*sqrt(length(ow.rv.p0) / length(ow.telstar.p0)), kwargs...)
    ow.om.rv.lm.M .= calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    result_rv, ow.om.rv.lm.s[:] = _OSW_optimize!(ow.rv, options)
    ow.o.rv .= rv_model(ow.om)
	recalc_total!(ow.o, d)

    if ignore_regularization
        copy_dict!(reg_tel_holder, ow.om.reg_tel)
        copy_dict!(reg_star_holder, ow.om.reg_star)
    end
    return result_telstar, result_rv
end

function train_OrderModel!(ow::OptimWorkspace, n::Int; kwargs...)
    for i in 1:n
        train_OrderModel!(ow; kwargs...)
    end
end

function fine_train_OrderModel!(ow::OptimWorkspace; g_tol::Real=_g_tol_def*sqrt(length(ow.telstar.p0)), kwargs...)
    train_OrderModel!(ow; kwargs...)  # 16s
    return train_OrderModel!(ow; g_tol=g_tol/10, f_tol=1e-8, kwargs...)
end
