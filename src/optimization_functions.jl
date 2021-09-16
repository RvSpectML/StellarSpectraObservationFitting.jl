# using LineSearches
using ParameterHandling
using Optim

function _loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::EXPRESData)
    y = (tel .* (star + rv)) - d.flux
    ans = 0
    Σ_thing = copy(d.Σ_lsf)
    for i in 1:size(y, 1)  # for each time
        yview = view(y, :, i)
        for j in 1:size(y, 2)  # for each pixel
            Σ_thing[:, j] = view(d.Σ_lsf, :, j) .* d.var[i, j]
        end
        ans += yview' * (Σ_thing \ yview)
    end
    return ans
end
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::GenericData) =
    sum((((tel .* (star + rv)) - d.flux) .^ 2) ./ d.var)


function loss(o::Output, om::OrderModel, d::Data;
    tel::LinearModel=om.tel, star::LinearModel=om.star, rv::LinearModel=om.rv)

    tel !== om.tel ? tel_o = interped_model(tel, om.lih_t2o) : tel_o = o.tel
    star !== om.star ? star_o = interped_model(star, om.lih_b2o) : star_o = o.star
    rv !== om.rv ? rv_o = interped_model(rv, om.lih_b2o) : rv_o = o.tel
    return _loss(tel_o, star_o, rv_o, d)
end

possible_θ = Union{LinearModel, AbstractVector, NamedTuple}

function loss_funcs(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    l_tel(tel::LinearModel) = loss(o, om, d; tel=tel) +
        model_prior(tel, om.reg_tel)
    function l_tel_s(tel::AbstractVector)
        lm = LinearModel(om.tel, tel)
        return loss(o, om, d; tel = lm) + model_prior(lm, om.reg_tel)
    end

    l_star(star::LinearModel) = loss(o, om, d; star = star) +
        model_prior(star, om.reg_star)
    function l_star_s(star::AbstractVector)
        lm = LinearModel(om.star, star)
        return loss(o, om, d; star = lm) + model_prior(lm, om.star_tel)
    end

    l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{LinearModel, LinearModel}}) =
        loss(o, om, d; tel = nt.tel, star = nt.star) +
        model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    function l_telstar_s(nt::NamedTuple{(:tel, :star,),<:Tuple{AbstractVector, AbstractVector}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        return loss(o, om, d; tel=tel, star=star) +
            model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv::AbstractVector) = loss(o, om, d; rv=BaseLinearModel(om.rv.M, rv))

    function l_total(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{LinearModel, LinearModel, AbstractVector}})
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, nt.star.μ), nt.rv)
        return loss(o, om, d; tel=nt.tel, star=nt.star, rv=rv) + model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    end
    function l_total_s(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{AbstractVector, AbstractVector, AbstractVector}})
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
    function l_telstar_s(star::AbstractVector)
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        return loss(o, om, d; tel=tel, star=star) +
            model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    l_rv(rv::AbstractVector) = loss(o, om, d; rv=BaseLinearModel(om.rv.M, rv))

    return l, l_telstar, l_telstar_s, l_rv
end

function loss_funcs_total(o::Output, om::OrderModel, d::Data)
    l() = loss(o, om, d)

    function l_total(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{LinearModel, LinearModel, AbstractVector}})
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, nt.star.μ), nt.rv)
        return loss(o, om, d; tel=nt.tel, star=nt.star, rv=rv) + model_prior(nt.tel, om.reg_tel) + model_prior(nt.star, om.reg_star)
    end
    function l_total_s(nt::NamedTuple{(:tel, :star, :rv,),<:Tuple{AbstractVector, AbstractVector, AbstractVector}})
        tel = LinearModel(om.tel, nt.tel)
        star = LinearModel(om.star, nt.star)
        rv = BaseLinearModel(calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.μ), nt.rv)
        return loss(o, om, d; tel=tel, star=star, rv=rv) + model_prior(tel, om.reg_tel) + model_prior(star, om.reg_star)
    end

    return l, l_total, l_total_s
end

function opt_funcs(loss::Function, pars::NamedTuple)
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
        return OptimSubWorkspace(θ, obj, opt, p0, unflatten)
    end
end

abstract type OptimWorkspace end

struct Workspace <: OptimWorkspace
    tel::OptimSubWorkspace
    star::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
    function Workspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, loss_tel, loss_tel_s, loss_star, loss_star_s, loss_rv = loss_funcs(o, om, d)
        rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true)
        if only_s
            tel = OptimSubWorkspace(om.tel.lm.s, loss_tel_s; use_cg=!only_s)
            star = OptimSubWorkspace(om.star.lm.s, loss_star_s; use_cg=!only_s)
        else
            tel = OptimSubWorkspace(om.tel.lm, loss_tel; use_cg=!only_s)
            star = OptimSubWorkspace(om.star.lm, loss_star; use_cg=!only_s)
        end
        ow = Workspace(tel, star, rv, om, o, d, only_s)
        if return_loss_f
            return ow, loss
        else
            return ow
        end
    end
    Workspace(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
        Workspace(om(inds), d(inds); kwargs...)
    Workspace(om::OrderModel, d::Data; kwargs...) =
        Workspace(om, Output(om), d; kwargs...)
end

struct WorkspaceTelStar <: OptimWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
    function WorkspaceTelStar(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, loss_telstar, loss_telstar_s, loss_rv = loss_funcs_telstar(o, om, d)
        rv = OptimSubWorkspace(om.rv.lm.s, loss_rv; use_cg=true)
        only_s ?
            telstar = OptimSubWorkspace((tel = om.tel.lm.s, star = om.star.lm.s,), loss_telstar_s; use_cg=!only_s) :
            telstar = OptimSubWorkspace((tel = om.tel.lm, star = om.star.lm,), loss_telstar; use_cg=!only_s)
        ow = WorkspaceTelStar(telstar, rv, om, o, d, only_s)
        if return_loss_f
            return ow, loss
        else
            return ow
        end
    end
    WorkspaceTelStar(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
        WorkspaceTelStar(om(inds), d(inds); kwargs...)
    WorkspaceTelStar(om::OrderModel, d::Data; kwargs...) =
        WorkspaceTelStar(om, Output(om), d; kwargs...)
end

struct WorkspaceTotal <: OptimWorkspace
    total::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    only_s::Bool
    function WorkspaceTotal(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, _, _, _, _, loss_opt = loss_funcs(o, om, d)
        total = OptimSubWorkspace(om.tel, om.star, om.rv, loss_opt, only_s)
        ow = WorkspaceTotal(total, om, o, d)
        if return_loss_f
            return ow, loss
        else
            return ow
        end
        loss, loss_total, loss_total_s = loss_funcs_total(o, om, d)
        only_s ?
            total = OptimSubWorkspace((tel = om.tel.lm.s, star = om.star.lm.s, rv = om.rv.lm.s), loss_total_s; use_cg=!only_s) :
            total = OptimSubWorkspace((tel = om.tel.lm, star = om.star.lm, rv=om.rv.lm.s), loss_total; use_cg=!only_s)
        ow = WorkspaceTotal(total, om, o, d, only_s)
        if return_loss_f
            return ow, loss
        else
            return ow
        end
    end
    WorkspaceTotal(om::OrderModel, d::Data, inds::AbstractVecOrMat; kwargs...) =
        WorkspaceTotal(om(inds), d(inds); kwargs...)
    WorkspaceTotal(om::OrderModel, d::Data; kwargs...) =
        WorkspaceTotal(om, Output(om), d; kwargs...)
end



function _OSW_optimize!(osw::OptimSubWorkspace, options::Optim.Options; return_result::Bool=true)
    result = Optim.optimize(osw.obj, osw.p0, osw.opt, options)
    osw.p0[:] = result.minimizer
    if return_result
        return result, unflatten(osw.p0)
    else
        return unflatten(osw.p0)
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

function train_OrderModel!(ow::Workspace; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def, kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)
    options = Optim.Options(;iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
    # optimize star
    if ow.only_s
        ow.om.star.lm.s[:] = _OSW_optimize!(ow.star, options; return_result=false)
    else
        copy_LinearModel!(_OSW_optimize!(ow.star, options; return_result=false), ow.om.star.lm)
    end
    ow.o.star[:, :] = star_model(ow.om)

    # optimize RVs
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    ow.om.rv.lm.s[:] = _OSW_optimize!(ow.rv, options; return_result=false)
    ow.o.rv[:, :] = rv_model(ow.om)

    # optimize tellurics
    if ow.only_s
        ow.om.tel.lm.s[:] = _OSW_optimize!(ow.tel, options; return_result=false)
    else
        copy_LinearModel!(_OSW_optimize!(ow.tel, options; return_result=false), ow.om.tel.lm)
    end
    ow.o.tel[:, :] = tel_model(ow.om)
end


function train_OrderModel!(ow::WorkspaceTelStar; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def*sqrt(length(ow.telstar.p0)), train_telstar::Bool=true, ignore_regularization::Bool=false, kwargs...)
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
        ow.o.star[:, :] = star_model(ow.om)
        ow.o.tel[:, :] = tel_model(ow.om)
    end

    # optimize RVs
    options = Optim.Options(;callback=optim_cb_local, g_tol=g_tol*sqrt(length(ow.rv.p0) / length(ow.telstar.p0)), kwargs...)
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    result_rv, ow.om.rv.lm.s[:] = _OSW_optimize!(ow.rv, options)
    ow.o.rv[:, :] = rv_model(ow.om)

    if ignore_regularization
        copy_dict!(reg_tel_holder, ow.om.reg_tel)
        copy_dict!(reg_star_holder, ow.om.reg_star)
    end
    return result_telstar, result_rv
end

function train_OrderModel!(ow::WorkspaceTotal; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def*sqrt(length(ow.total.p0)), kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)
    options = Optim.Options(;iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
    # optimize tellurics and star
    result, nt = _OSW_optimize!(ow.total, options)
    if ow.only_s
        _custom_copy!(nt, ow.om.tel.lm.s, ow.om.star.lm.s, ow.om.rv.lm.s)
    else
        _custom_copy!(nt, ow.om.tel.lm, ow.om.star.lm, ow.om.rv.lm.s)
    end
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    ow.o.star[:, :] = star_model(ow.om)
    ow.o.tel[:, :] = tel_model(ow.om)
    ow.o.rv[:, :] = rv_model(ow.om)
    return result
end

function train_OrderModel!(ow::OptimWorkspace, n::Int; kwargs...)
    for i in 1:n
        train_OrderModel!(ow; kwargs...)
    end
end

train_OrderModel!(ow::WorkspaceTotal, iterations::Int; kwargs...) =
    train_OrderModel!(ow; iterations=iterations, kwargs...)

function fine_train_OrderModel!(ow::OptimWorkspace; g_tol::Real=_g_tol_def*sqrt(length(ow.telstar.p0)), kwargs...)
    train_OrderModel!(ow; kwargs...)  # 16s
    return train_OrderModel!(ow; g_tol=g_tol/10, f_tol=1e-8, kwargs...)
end
