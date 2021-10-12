# using LineSearches
using ParameterHandling
using Optim
using Zygote

# χ² loss function
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::GenericData) =
    sum(((total_model(tel, star, rv) .- d.flux) .^ 2) ./ d.var)
# χ² loss function broadened by an lsf at each time (about 2x slower)
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::LSFData) =
    mapreduce(i -> sum((((d.lsf_broadener[i] * (view(tel, :, i) .* (view(star, :, i) .+ view(rv, :, i)))) .- view(d.flux, :, i)) .^ 2) ./ view(d.var, :, i)), +, 1:size(tel, 2))

function loss(o::Output, om::OrderModel, d::Data;
	tel::LinearModel=om.tel.lm, star::LinearModel=om.star.lm, rv::LinearModel=om.rv.lm,
	recalc_tel::Bool=true, recalc_star::Bool=true, recalc_rv::Bool=true)

    recalc_tel ? tel_o = tel_model(om; lm=tel) : tel_o = o.tel
    recalc_star ? star_o = star_model(om; lm=star) : star_o = o.star
    recalc_rv ? rv_o = rv_model(om; lm=rv) : rv_o = o.rv
    return _loss(tel_o, star_o, rv_o, d)
end

possible_θ = Union{LinearModel, AbstractMatrix, NamedTuple}

function loss_funcs_telstar(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{LinearModel, LinearModel}}) =
        loss(o, om, d; tel=nt.tel, star=nt.star, recalc_rv=false) +
        model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    function l_telstar_s(nt::NamedTuple{(:tel, :star,),<:Tuple{AbstractVector, AbstractVector}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        return loss(o, om, d; tel=tel, star=star, recalc_rv=false) +
            model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv::AbstractMatrix) = loss(o, om, d; rv=BaseLinearModel(om.rv.M, rv), recalc_tel=false, recalc_star=false)

    return l, l_telstar, l_telstar_s, l_rv
end


function opt_funcs(loss::Function, pars::possible_θ)
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns NamedTuple of untransformed params
    f = loss ∘ unflatten
    function g!(G, θ)
        G .= only(Zygote.gradient(f, θ))
    end
    function fg_obj!(G, θ)
        l, back = Zygote.pullback(f, θ)
        G .= only(back(1))
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
    function OptimSubWorkspace(θ::possible_θ, loss::Function; use_cg::Bool=true)
        p0, obj, unflatten = opt_funcs(loss, θ)
        # opt = LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=NaN))
        use_cg ? opt = ConjugateGradient() : opt = LBFGS()
        # initial_state(method::LBFGS, ...) doesn't use the options for anything
        return new(θ, obj, opt, p0, unflatten)
    end
end

struct OptimWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
    function OptimWorkspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, loss_telstar, loss_telstar_s, loss_rv = loss_funcs_telstar(o, om, d)
        rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true)
        only_s ?
            telstar = OptimSubWorkspace((tel = om.tel.lm.s, star = om.star.lm.s,), loss_telstar_s; use_cg=!only_s) :
            telstar = OptimSubWorkspace((tel = om.tel.lm, star = om.star.lm,), loss_telstar; use_cg=!only_s)
        ow = new(telstar, rv, om, o, d, only_s)
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
end

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
