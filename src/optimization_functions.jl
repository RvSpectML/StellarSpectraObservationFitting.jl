using LineSearches
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, tfd::Data) =
    sum((((tel .* (star + rv)) - tfd.flux) .^ 2) ./ tfd.var)
loss(tfo::Output, tfd::Data) = _loss(tfo.tel, tfo.star, tfo.rv, tfd)
loss(tfom::OrderModel, tfd::Data) = _loss(tel_model(tfom), star_model(tfom), rv_model(tfom), tfd)
loss_tel(tfo::Output, tfom::OrderModel, tfd::Data) = _loss(tel_model(tfom), tfo.star, tfo.rv, tfd) + tel_prior(tfom)
loss_star(tfo::Output, tfom::OrderModel, tfd::Data) = _loss(tfo.tel, star_model(tfom), tfo.rv, tfd) + star_prior(tfom)
loss_telstar(tfo::Output, tfom::OrderModel, tfd::Data) = _loss(tel_model(tfom), star_model(tfom), tfo.rv, tfd) + star_prior(tfom) + tel_prior(tfom)
loss_rv(tfo::Output, tfom::OrderModel, tfd::Data) = _loss(tfo.tel, tfo.star, rv_model(tfom), tfd)
loss_opt(tfom::OrderModel, tfd::Data) =
    _loss(tel_model(tfom),
        star_model(tfom),
        spectra_interp(
            _eval_blm(
                calc_doppler_component_RVSKL_Flux(tfom.star.λ, tfom.star.lm.μ),
                tfom.rv.lm.s),
            tfom.lih_b2o),
        tfd) + star_prior(tfom) + tel_prior(tfom)
function loss_funcs(tfo::Output, tfom::OrderModel, tfd::Data)
    l() = loss(tfo, tfd)
    l_tel() = loss_tel(tfo, tfom, tfd)
    l_star() = loss_star(tfo, tfom, tfd)
    l_telstar() = loss_telstar(tfo, tfom, tfd)
    l_rv() = loss_rv(tfo, tfom, tfd)
    l_opt() = loss_opt(tfom, tfd)
    return l, l_tel, l_star, l_telstar, l_rv, l_opt
end


struct OptimSubWorkspace
    θ::Flux.Params
    obj::OnceDifferentiable
    opt::Optim.FirstOrderOptimizer
    p0::Vector
    function OptimSubWorkspace(θ::Flux.Params, loss::Function; use_cg::Bool=true)
        _, _, _, p0, obj = opt_funcs(loss, θ)
        # opt = LBFGS(alphaguess = LineSearches.InitialHagerZhang(α0=NaN))
        use_cg ? opt = ConjugateGradient() : opt = LBFGS()
        # initial_state(method::LBFGS, ...) doesn't use the options for anything
        return OptimSubWorkspace(θ, obj, opt, p0)
    end
    function OptimSubWorkspace(tfsm::Submodel, loss::Function, only_s::Bool)
        if only_s
            θ = Flux.params(tfsm.lm.s)
        else
            θ = Flux.params(tfsm.lm.M, tfsm.lm.s, tfsm.lm.μ)
        end
        return OptimSubWorkspace(θ, loss; use_cg=!only_s)
    end
    function OptimSubWorkspace(tfsm1::Submodel, tfsm2::Submodel, loss::Function, only_s::Bool)
        if only_s
            θ = Flux.params(tfsm1.lm.s, tfsm2.lm.s)
        else
            θ = Flux.params(tfsm1.lm.M, tfsm1.lm.s, tfsm1.lm.μ, tfsm2.lm.M, tfsm2.lm.s, tfsm2.lm.μ)
        end
        return OptimSubWorkspace(θ, loss; use_cg=!only_s)
    end
    function OptimSubWorkspace(tfsm1::Submodel, tfsm2::Submodel, tfsm3::Submodel, loss::Function, only_s::Bool)
        if only_s
            θ = Flux.params(tfsm1.lm.s, tfsm2.lm.s, tfsm3.lm.s)
        else
            θ = Flux.params(tfsm1.lm.M, tfsm1.lm.s, tfsm1.lm.μ, tfsm2.lm.M, tfsm2.lm.s, tfsm2.lm.μ, tfsm3.lm.s)
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
    tfom::OrderModel
    tfo::Output
    tfd::Data
    function Workspace(tfom::OrderModel, tfo::Output, tfd::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, loss_tel, loss_star, _, loss_rv, _ = loss_funcs(tfo, tfom, tfd)
        tel = OptimSubWorkspace(tfom.tel, loss_tel, only_s)
        star = OptimSubWorkspace(tfom.star, loss_star, only_s)
        rv = OptimSubWorkspace(tfom.rv, loss_rv, true)
        tfow = Workspace(tel, star, rv, tfom, tfo, tfd)
        if return_loss_f
            return tfow, loss
        else
            return tfow
        end
    end
    Workspace(tfom::OrderModel, tfd::Data, inds::AbstractVecOrMat; kwargs...) =
        Workspace(tfom(inds), tfd(inds); kwargs...)
    Workspace(tfom::OrderModel, tfd::Data; kwargs...) =
        Workspace(tfom, Output(tfom), tfd; kwargs...)
    function Workspace(tel, star, rv, tfom, tfo, tfd)
        @assert length(tel.θ) == length(star.θ)
        @assert (length(tel.θ) == 1) || (length(tel.θ) == 3)
        @assert length(rv.θ) == 1
        new(tel, star, rv, tfom, tfo, tfd)
    end
end

struct WorkspaceTelStar <: OptimWorkspace
    telstar::OptimSubWorkspace
    rv::OptimSubWorkspace
    tfom::OrderModel
    tfo::Output
    tfd::Data
    function WorkspaceTelStar(tfom::OrderModel, tfo::Output, tfd::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, _, _, loss_telstar, loss_rv, _ = loss_funcs(tfo, tfom, tfd)
        telstar = OptimSubWorkspace(tfom.tel, tfom.star, loss_telstar, only_s)
        rv = OptimSubWorkspace(tfom.rv, loss_rv, true)
        tfow = WorkspaceTelStar(telstar, rv, tfom, tfo, tfd)
        if return_loss_f
            return tfow, loss
        else
            return tfow
        end
    end
    WorkspaceTelStar(tfom::OrderModel, tfd::Data, inds::AbstractVecOrMat; kwargs...) =
        WorkspaceTelStar(tfom(inds), tfd(inds); kwargs...)
    WorkspaceTelStar(tfom::OrderModel, tfd::Data; kwargs...) =
        WorkspaceTelStar(tfom, Output(tfom), tfd; kwargs...)
    function WorkspaceTelStar(telstar::OptimSubWorkspace,
        rv::OptimSubWorkspace,
        tfom::OrderModel,
        tfo::Output,
        tfd::Data)

        @assert (length(telstar.θ) == 2) || (length(telstar.θ) == 6)
        @assert length(rv.θ) == 1
        new(telstar, rv, tfom, tfo, tfd)
    end
    function WorkspaceTelStar(tfws::WorkspaceTelStar)
        WorkspaceTelStar(tfws.telstar, rv, tfom, tfo, tfd)
    end
end

struct WorkspaceTotal <: OptimWorkspace
    total::OptimSubWorkspace
    tfom::OrderModel
    tfo::Output
    tfd::Data
    function WorkspaceTotal(tfom::OrderModel, tfo::Output, tfd::Data; return_loss_f::Bool=false, only_s::Bool=false)
        loss, _, _, _, _, loss_opt = loss_funcs(tfo, tfom, tfd)
        total = OptimSubWorkspace(tfom.tel, tfom.star, tfom.rv, loss_opt, only_s)
        tfow = WorkspaceTotal(total, tfom, tfo, tfd)
        if return_loss_f
            return tfow, loss
        else
            return tfow
        end
    end
    WorkspaceTotal(tfom::OrderModel, tfd::Data, inds::AbstractVecOrMat; kwargs...) =
        WorkspaceTotal(tfom(inds), tfd(inds); kwargs...)
    WorkspaceTotal(tfom::OrderModel, tfd::Data; kwargs...) =
        WorkspaceTotal(tfom, Output(tfom), tfd; kwargs...)
    function WorkspaceTotal(total, tfom, tfo, tfd)
        @assert (length(total.θ) == 3) || (length(total.θ) == 7)
        new(total, tfom, tfo, tfd)
    end
end

function _Flux_optimize!(θ::Flux.Params, obj::OnceDifferentiable, p0::Vector,
    opt::Optim.FirstOrderOptimizer, options::Optim.Options)
    result = Optim.optimize(obj, p0, opt, options)
    copyto!(p0, θ)
    return result
end
_Flux_optimize!(tfosw::OptimSubWorkspace, options::Optim.Options) =
    _Flux_optimize!(tfosw.θ, tfosw.obj, tfosw.p0, tfosw.opt, options)


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

function train_OrderModel!(tfow::Workspace; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def, kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)
    options = Optim.Options(iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
    # optimize star
    _Flux_optimize!(tfow.star, options)
    tfow.tfo.star[:, :] = star_model(tfow.tfom)

    # optimize RVs
    tfow.tfom.rv.lm.M[:] = calc_doppler_component_RVSKL(tfow.tfom.star.λ, tfow.tfom.star.lm.μ)
    _Flux_optimize!(tfow.rv, options)
    tfow.tfo.rv[:, :] = rv_model(tfow.tfom)

    # optimize tellurics
    _Flux_optimize!(tfow.tel, options)
    tfow.tfo.tel[:, :] = tel_model(tfow.tfom)
end

function train_OrderModel!(tfow::WorkspaceTelStar; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def*sqrt(length(tfow.telstar.p0)), train_telstar::Bool=true, ignore_regularization::Bool=false, kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)

    if ignore_regularization
        reg_tel_holder = copy(tfow.tfom.reg_tel)
        reg_star_holder = copy(tfow.tfom.reg_star)
        zero_regularization(tfow.tfom)
    end

    if train_telstar
        options = Optim.Options(iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
        # optimize tellurics and star
        result_telstar = _Flux_optimize!(tfow.telstar, options)
        tfow.tfo.star[:, :] = star_model(tfow.tfom)
        tfow.tfo.tel[:, :] = tel_model(tfow.tfom)
    end

    # optimize RVs
    options = Optim.Options(callback=optim_cb_local, g_tol=g_tol*sqrt(length(tfow.rv.p0) / length(tfow.telstar.p0)), kwargs...)
    tfow.tfom.rv.lm.M[:] = calc_doppler_component_RVSKL(tfow.tfom.star.λ, tfow.tfom.star.lm.μ)
    result_rv = _Flux_optimize!(tfow.rv, options)
    tfow.tfo.rv[:, :] = rv_model(tfow.tfom)

    if ignore_regularization
        copy_dict!(reg_tel_holder, tfow.tfom.reg_tel)
        copy_dict!(reg_star_holder, tfow.tfom.reg_star)
    end
    return result_telstar, result_rv
end

function train_OrderModel!(tfow::WorkspaceTotal; print_stuff::Bool=_print_stuff_def, iterations::Int=_iter_def, f_tol::Real=_f_tol_def, g_tol::Real=_g_tol_def*sqrt(length(tfow.total.p0)), kwargs...)
    optim_cb_local(x::OptimizationState) = optim_cb(x; print_stuff=print_stuff)
    options = Optim.Options(iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local, kwargs...)
    # optimize tellurics and star
    result = _Flux_optimize!(tfow.total, options)
    tfow.tfom.rv.lm.M[:] = calc_doppler_component_RVSKL(tfow.tfom.star.λ, tfow.tfom.star.lm.μ)
    tfow.tfo.star[:, :] = star_model(tfow.tfom)
    tfow.tfo.tel[:, :] = tel_model(tfow.tfom)
    tfow.tfo.rv[:, :] = rv_model(tfow.tfom)
    return result
end

function train_OrderModel!(tfow::OptimWorkspace, n::Int; kwargs...)
    for i in 1:n
        train_OrderModel!(tfow; kwargs...)
    end
end

train_OrderModel!(tfow::WorkspaceTotal, iterations::Int; kwargs...) =
    train_OrderModel!(tfow; iterations=iterations, kwargs...)
