using LineSearches
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
loss(o::Output, d::Data) = _loss(o.tel, o.star, o.rv, d)
loss(om::OrderModel, d::Data) = _loss(tel_model(om), star_model(om), rv_model(om), d)
loss_tel(o::Output, om::OrderModel, d::Data) = _loss(tel_model(om), o.star, o.rv, d) + tel_prior(om)
loss_star(o::Output, om::OrderModel, d::Data) = _loss(o.tel, star_model(om), o.rv, d) + star_prior(om)
loss_telstar(o::Output, om::OrderModel, d::Data) = _loss(tel_model(om), star_model(om), o.rv, d) + star_prior(om) + tel_prior(om)
loss_rv(o::Output, om::OrderModel, d::Data) = _loss(o.tel, o.star, rv_model(om), d)
loss_opt(om::OrderModel, d::Data) =
    _loss(tel_model(om),
        star_model(om),
        spectra_interp(
            _eval_blm(
                calc_doppler_component_RVSKL_Flux(om.star.λ, om.star.lm.μ),
                om.rv.lm.s),
            om.lih_b2o),
        d) + star_prior(om) + tel_prior(om)
function loss_funcs(o::Output, om::OrderModel, d::Data)
    l() = loss(o, d)
    _loss(tel, star, rv, d)
    l_tel(nt::NamedTuple) = loss_tel(o, om, d)
    l_star(nt::NamedTuple) = loss_star(o, om, d)
    l_telstar(nt::NamedTuple) = loss_telstar(o, om, d)
    l_rv(nt::NamedTuple) = loss_rv(o, om, d)
    l_opt(nt::NamedTuple) = loss_opt(om, d)
    return l, l_tel, l_star, l_telstar, l_rv, l_opt
end

function loss_funcs(o::Output, om::OrderModel, d::Data)
    l() = loss(o, d)
    l_tel() = loss_tel(o, om, d)
    l_star() = loss_star(o, om, d)
    l_telstar() = loss_telstar(o, om, d)
    l_rv() = loss_rv(o, om, d)
    l_opt() = loss_opt(om, d)
    return l, l_tel, l_star, l_telstar, l_rv, l_opt
end

function opt_funcs(loss::Function, pars::NamedTuple)
    flat_initial_params, unflatten = value_flatten(pars)  # unflatten returns NamedTuple of untransformed params
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
    θ::NamedTuple
    obj::OnceDifferentiable
    opt::Optim.FirstOrderOptimizer
    p0::Vector
    function OptimSubWorkspace(θ::NamedTuple, loss::Function; use_cg::Bool=true)
        _, _, _, p0, obj = opt_funcs(loss, θ)
        # opt = LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=NaN))
        use_cg ? opt = ConjugateGradient() : opt = LBFGS()
        # initial_state(method::LBFGS, ...) doesn't use the options for anything
        return OptimSubWorkspace(θ, obj, opt, p0)
    end
    function OptimSubWorkspace(sm::Submodel, loss::Function, only_s::Bool)
        if only_s
            θ = (s = sm.lm.s)
        else
            θ = (M = sm.lm.M, s = sm.lm.s, μ = sm.lm.μ)
        end
        return OptimSubWorkspace(θ, loss; use_cg=!only_s)
    end
    function OptimSubWorkspace(sm1::Submodel, sm2::Submodel, loss::Function, only_s::Bool)
        T1 = typeof(sm1.lm) <: TemplateModel
        T2 = typeof(sm2.lm) <: TemplateModel
        if only_s
            if T1
                θ = Flux.params(s = sm2.lm.s)
            elseif T2
                θ = Flux.params(s = sm1.lm.s)
            else
                θ = Flux.params(s1 = sm1.lm.s, s2 = sm2.lm.s)
            end
        else
            if T1 && T2
                θ = Flux.params(μ1 = sm1.lm.μ, μ2 = sm2.lm.μ)
            elseif T1
                θ = Flux.params(μ1 = sm1.lm.μ, M2 = sm2.lm.M, s2 = sm2.lm.s, μ2 = sm2.lm.μ)
            elseif T2
                θ = Flux.params(sm1.lm.M, sm1.lm.s, sm1.lm.μ, sm2.lm.μ)
            else
                θ = Flux.params(sm1.lm.M, sm1.lm.s, sm1.lm.μ, sm2.lm.M, sm2.lm.s, sm2.lm.μ)
            end
        end
        return OptimSubWorkspace(θ, loss; use_cg=!only_s)
    end
    function OptimSubWorkspace(sm1::Submodel, sm2::Submodel, sm3::Submodel, loss::Function, only_s::Bool)
        if only_s
            θ = Flux.params(sm1.lm.s, sm2.lm.s, sm3.lm.s)
        else
            θ = Flux.params(sm1.lm.M, sm1.lm.s, sm1.lm.μ, sm2.lm.M, sm2.lm.s, sm2.lm.μ, sm3.lm.s)
        end
        return OptimSubWorkspace(θ, loss; use_cg=!only_s)
    end
    function OptimSubWorkspace(θ, obj, opt, p0)
        len = 0
        for i in 1:length(θ)
            len += length(θ[i])
        end
        @assert len == length(p0)
        new(θ, obj, opt, p0)
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
    function Workspace(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, loss_tel, loss_star, _, loss_rv, _ = loss_funcs(o, om, d)
        tel = OptimSubWorkspace(om.tel, loss_tel, only_s)
        star = OptimSubWorkspace(om.star, loss_star, only_s)
        rv = OptimSubWorkspace(om.rv, loss_rv, true)
        ow = Workspace(tel, star, rv, om, o, d)
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
    function Workspace(tel, star, rv, om, o, d)
        @assert length(tel.θ) == length(star.θ)
        @assert (length(tel.θ) == 1) || (length(tel.θ) == 3)
        @assert length(rv.θ) == 1
        new(tel, star, rv, om, o, d)
    end
end

struct WorkspaceTelStar <: OptimWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    function WorkspaceTelStar(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, _, _, loss_telstar, loss_rv, _ = loss_funcs(o, om, d)
        telstar = OptimSubWorkspace(om.tel, om.star, loss_telstar, only_s)
        rv = OptimSubWorkspace(om.rv, loss_rv, true)
        ow = WorkspaceTelStar(telstar, rv, om, o, d)
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
    function WorkspaceTelStar(telstar::OptimSubWorkspace,
        rv::OptimSubWorkspace,
        om::OrderModel,
        o::Output,
        d::Data)

        # @assert (length(telstar.θ) == 2) || (length(telstar.θ) == 6)
        @assert length(rv.θ) == 1
        new(telstar, rv, om, o, d)
    end
end

struct WorkspaceTotal <: OptimWorkspace
    total::OptimSubWorkspace
    om::OrderModel
    o::Output
    d::Data
    function WorkspaceTotal(om::OrderModel, o::Output, d::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, _, _, _, _, loss_opt = loss_funcs(o, om, d)
        total = OptimSubWorkspace(om.tel, om.star, om.rv, loss_opt, only_s)
        ow = WorkspaceTotal(total, om, o, d)
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
    function WorkspaceTotal(total, om, o, d)
        @assert (length(total.θ) == 3) || (length(total.θ) == 7)
        new(total, om, o, d)
    end
end

function _Flux_optimize!(θ::NamedTuple, obj::OnceDifferentiable, p0::Vector,
    opt::Optim.FirstOrderOptimizer, options::Optim.Options)
    result = Optim.optimize(obj, p0, opt, options)
    copyto!(p0, θ)
    return result
end
_Flux_optimize!(osw::OptimSubWorkspace, options::Optim.Options) =
    _Flux_optimize!(osw.θ, osw.obj, osw.p0, osw.opt, options)


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

function train_OrderModel!(ow::Workspace; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def, kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)
    options = Optim.Options(;iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
    # optimize star
    _Flux_optimize!(ow.star, options)
    ow.o.star[:, :] = star_model(ow.om)

    # optimize RVs
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    _Flux_optimize!(ow.rv, options)
    ow.o.rv[:, :] = rv_model(ow.om)

    # optimize tellurics
    _Flux_optimize!(ow.tel, options)
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
        result_telstar = _Flux_optimize!(ow.telstar, options)
        ow.o.star[:, :] = star_model(ow.om)
        ow.o.tel[:, :] = tel_model(ow.om)
    end

    # optimize RVs
    options = Optim.Options(;callback=optim_cb_local, g_tol=g_tol*sqrt(length(ow.rv.p0) / length(ow.telstar.p0)), kwargs...)
    ow.om.rv.lm.M[:] = calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
    result_rv = _Flux_optimize!(ow.rv, options)
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
    result = _Flux_optimize!(ow.total, options)
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
