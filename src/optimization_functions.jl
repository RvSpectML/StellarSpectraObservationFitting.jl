# using LineSearches
using ParameterHandling
using Optim
using Zygote


_χ2_loss(model::AbstractMatrix, d::Data) =
    sum(((model - d.flux) .^ 2) ./ d.var)
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::GenericData) =
    _χ2_loss(tel .* (star + rv), d)
function _loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::LSFData)
    model = tel .* (star + rv)
    for i in 1:size(model, 2)
        model[:, i] = d.lsf_broadener[i] * model[:, i]
    end
    return _χ2_loss(model, d)
end

function loss(o::Output, om::OrderModel, d::Data;
    tel::LinearModel=om.tel.lm, star::LinearModel=om.star.lm, rv::LinearModel=om.rv.lm)

    tel !== om.tel ? tel_o = tel_model(om, d) : tel_o = o.tel
    star !== om.star ? star_o = star_model(om, d) : star_o = o.star
    rv !== om.rv ? rv_o = rv_model(om, d) : rv_o = o.tel
    return _loss(tel_o, star_o, rv_o, d)
end

possible_θ = Union{LinearModel, AbstractMatrix, NamedTuple}

function loss_funcs(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    l_tel(tel::LinearModel) = loss(o, om, d; tel=tel) +
        model_prior(tel, om.reg_tel)
    function l_tel_s(tel::AbstractMatrix)
        lm = LinearModel(om.tel, tel)
        return loss(o, om, d; tel = lm) + model_prior(lm, om.reg_tel)
    end

    l_star(star::LinearModel) = loss(o, om, d; star = star) +
        model_prior(star, om.reg_star)
    function l_star_s(star::AbstractMatrix)
        lm = LinearModel(om.star, star)
        return loss(o, om, d; star = lm) + model_prior(lm, om.star_tel)
    end

    l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{LinearModel, LinearModel}}) =
        loss(o, om, d; tel = nt.tel, star = nt.star) +
        model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    function l_telstar_s(nt::NamedTuple{(:tel, :star,),<:Tuple{AbstractMatrix, AbstractMatrix}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        return loss(o, om, d; tel=tel, star=star) +
            model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv::AbstractMatrix) = loss(o, om, d; rv=BaseLinearModel(om.rv.M, rv))

    function l_total(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{LinearModel, LinearModel, AbstractMatrix}})
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, nt.star.μ), nt.rv)
        return loss(o, om, d; tel=nt.tel, star=nt.star, rv=rv) + model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    end
    function l_total_s(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.μ), nt.rv)
        return loss(o, om, d; tel=tel, star=star, rv=rv) + model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    return l, l_tel, l_tel_s, l_star, l_star_s, l_rv
end

function loss_funcs_telstar(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{LinearModel, LinearModel}}) =
        loss(o, om, d; tel = nt.tel, star = nt.star) +
        model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    function l_telstar_s(star::AbstractMatrix)
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        return loss(o, om, d; tel=tel, star=star) +
            model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv::AbstractMatrix) = loss(o, om, d; rv=BaseLinearModel(om.rv.M, rv))

    return l, l_telstar, l_telstar_s, l_rv
end

function loss_funcs_total(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    function l_total(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{LinearModel, LinearModel, AbstractMatrix}})
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, nt.star.μ), nt.rv)
        return loss(o, om, d; tel=nt.tel, star=nt.star, rv=rv) + model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    end
    function l_total_s(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{AbstractMatrix, AbstractMatrix, AbstractMatrix}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.μ), nt.rv)
        return loss(o, om, d; tel=tel, star=star, rv=rv) + model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    return l, l_total, l_total_s
end

function opt_funcs(loss::Function, pars::possible_θ)
    flat_initial_params, unflatten = flatten(pars)  # unflatten returns NamedTuple of untransformed params
    f = loss ∘ unflatten
    function g!(G, θ)
        G[:] = only(Zygote.gradient(f, θ))
    end
    function fg_obj!(G, θ)
        l, back = Zygote.pullback(f, θ)
        G[:] = only(back(1))
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
    osw.p0[:] = result.minimizer
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
        if typeof(to)<:LinearModel
            copy_LinearModel!(from[k], to[i])
        elseif typeof(to)<:AbstractVector
            to[:] = from[k]
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
        ow.o.star[:, :] = star_model(ow.om, d)
        ow.o.tel[:, :] = tel_model(ow.om, d)
    end

    # optimize RVs
    options = Optim.Options(;callback=optim_cb_local, g_tol=g_tol*sqrt(length(ow.rv.p0) / length(ow.telstar.p0)), kwargs...)
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    result_rv, ow.om.rv.lm.s[:] = _OSW_optimize!(ow.rv, options)
    ow.o.rv[:, :] = rv_model(ow.om, d)

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
