## Importing packages
using Pkg
Pkg.activate("EXPRES")

import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using JLD2
using Statistics
import StatsBase

## Setting up necessary variables

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 2)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 68)  # 68 has a bunch of tels, 47 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 3)

## Loading in data and initializing model
save_path = expres_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end
if which_opt == 1
    save_path *= "optim_"
end

if isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    end
else
    model_upscale = 2 * sqrt(2)
    @time model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8, upscale=model_upscale)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
    if !use_reg
        SSOF.zero_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

using AbstractGPs, KernelFunctions, TemporalGPs
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using LinearAlgebra
using SparseArrays
using BenchmarkTools

## Looking into using sparse inverse of covariance
x = model.tel.log_λ
x2 = x[1:1000]
y = model.tel.lm.μ
y2 = y[1:1000]
f2 = SSOF.SOAP_gp
f3 = GP(SSOF.SOAP_gp_params.var_kernel * PiecewisePolynomialKernel(;degree=3, dim=1) ∘ ScaleTransform(SSOF.SOAP_gp_params.λ/4))

# f = to_sde(f3, SArrayStorage(Float64))
# Σ = cov(f3(x2)) + diagm(1e-3 .* ones(length(x2)))
# Σs = dropzeros!(sparse(Σ))
# # Σs = droptol!(sparse(Σ), 1e-6)
# cholΣ = cholesky(Σs)
# # luΣ = lu(Σs)
# # qrΣ = qr(Σs)
# # invΣ = inv(cholΣ)
# # invΣ = cholΣ \ LinearAlgebra.I(length(y))
# # invΣ = inv(Σ)
#
#
# -(dot(y2, (cholΣ \ y2)) + logdet(cholΣ) + length(y2) * log(2*π))/2
# logpdf(f2(x2), y2)
# @btime logpdf(f2(x2), y2)
# @btime dot(y2, (cholΣ \ y2))
# # @time dot(y, (luΣ\y))
# # @time dot(y, (qrΣ\y))
# # @time dot(y, (invΣ*y))


## Looking into using a gutted version of TemporalGPs attempt 2
# LTI SDE : Linear Time Invariant Stochastic differential equation
# LGGSM : linear-Gaussian state-space model
# LGC : Linear-Gaussian Conditional
# SDE : Stochastic differential equation
import TemporalGPs; TGP = TemporalGPs
using Zygote

fx

fx = f2(x)
@time logpdf(fx, y)
@profiler for i in 1:100
    logpdf(fx, y)
end

# @enter AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real})
ft = fx
TGP._logpdf(ft, y)

# @enter function _logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
# logpdf(TGP.build_lgssm(ft), TGP.observations_to_time_form(ft.x, y))
model2 = TGP.build_lgssm(ft)
@time logpdf(model2, y)

#@enter function AbstractGPs.logpdf(model::LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}})
# sum(scan_emit(step_logpdf, zip(model, y), x0(model), eachindex(model))[1])
sum(TGP.scan_emit(TGP.step_logpdf, zip(model2, y), model2.transitions.x0, eachindex(model2))[1])

#@enter function scan_emit(f, xs, state, idx)
state = model2.transitions.x0
xs = zip(model2, y)
idx = eachindex(model2)

(lml, state) = TGP.step_logpdf(state, TGP._getindex(xs, idx[1]))
ms = [state.m]
lmls = Vector{typeof(yy)}(undef, length(idx))
lmls[idx[1]] = lml
for t in idx[2:end]
    (yy, state) = TGP.step_logpdf(state, TGP._getindex(xs, t))
    append!(ms, [state.m])
    lmls[t] = yy
end
lmls

plot([m[1] for m in ms])

TGP._getindex(xs, idx[1])
#@enter step_logpdf(x::Gaussian, (model, y))
xx = state

y[1] == yyy
model3, yyy = TGP._getindex(xs, idx[1])
xp = TGP.predict(xx, TGP.transition_dynamics(model3))
model4 = TGP.emission_dynamics(model3)
xf, lml = TGP.posterior_and_lml(xp, TGP.emission_dynamics(model3), yyy)

model3.emission
#@enter function posterior_and_lml(x::Gaussian, f::ScalarOutputLGC, y::T) where {T<:Real}

m, P = xp.m, xp.P
A, a, Q = model4.A, model4.a, model4.Q
V = A * xp.P
sqrtS = sqrt(V * A' + Q)  # std of obs at time
B = sqrtS \ V

sqrtS \ (yyy - (A * m + a))
(yyy - (A * m + a)) / sqrtS

α = sqrtS \ (yyy - (A * m + a))
lml = -(log(2π) + 2 * log(sqrtS) + α^2) / 2
return TGP.Gaussian(m + B'α, P - B'B), lml

#@enter function predict(x::Gaussian, f::AbstractLGC)
    A, a, Q = get_fields(f)
    m, P = get_fields(x)
    # Symmetric wrapper needed for numerical stability. Do not unwrap.
    return Gaussian(A * m + a, A * symmetric(P) * A' + Q)
end

#@enter function build_lgssm(ft::FiniteLTISDE)
# ftt = Zygote.literal_getfield(ft, Val(:f))
# x = Zygote.literal_getfield(ft, Val(:x))
# Σlmls = TGP.noise_var_to_time_form(x, Zygote.literal_getfield(ft, Val(:Σy)))
# TGP.build_lgssm(ftt, x, Σlmls)
TGP.build_lgssm(ft.f, ft.x, diag(ft.Σy))

ft.f.f.kernel
#@enter function build_lgssm(f::LTISDE, x::AbstractVector, Σlmls::AbstractVector)
# k = TGP.get_kernel(f)
# s = Zygote.literal_getfield(f, Val(:storage))
# As, as, Qs, emission_proj, x0 = lgssm_components(k, x, s)
# LGSSM(GaussMarkovModel(Forward(), As, as, Qs, x0), build_emissions(emission_proj, Σlmls))
As, as, Qs, emission_proj, x0 = TGP.lgssm_components(ft.f.f.kernel, ft.x, ft.f.storage)
TGP.LGSSM(TGP.GaussMarkovModel(TGP.Forward(), As, as, Qs, x0), TGP.build_emissions(emission_proj, diag(ft.Σy)))
## actually tring to build my version
import TemporalGPs; TGP = TemporalGPs
using Zygote
using LinearAlgebra
fx = f2(x)
@time logpdf(fx, y)
my_logpdf2(ft::TGP.FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}}) =
    my_logpdf2(TGP.build_lgssm(ft), y)
@time my_logpdf2(fx, y)

lgssm = TGP.build_lgssm(fx)
@time my_logpdf2(lgssm, y)
@time my_logpdf3(lgssm, y)

hmm = zip(lgssm, y)
hmm.is[1][1]
my_logpdf2(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(TGP.scan_emit(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])
my_logpdf3(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(my_scan_emit2(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])

@profiler for i in 1:10
    my_logpdf3(lggsm, y)
end

my_logpdf3(lggsm, y)
length(y) * 7
model2.transitions.x0
function my_scan_emit2(f::Function, xs::Iterators.Zip, state::TGP.Gaussian, idx::UnitRange)
    (yy, state) = f(state, TGP._getindex(xs, idx[1]))
    ys = Vector{typeof(yy)}(undef, length(idx))
    ys[idx[1]] = yy

    for t in idx[2:end]
        (yy, state) = f(state, TGP._getindex(xs, t))
        ys[t] = yy
    end

    return (ys, state)
end

function my_scan_emit2(f::Function, xs::Iterators.Zip, state::TGP.Gaussian, idx::UnitRange)
    (yy, state) = f(state, TGP._getindex(xs, idx[1]))
    ys = Vector{typeof(yy)}(undef, length(idx))
    ys[idx[1]] = yy

    for t in idx[2:end]
        (yy, state) = f(state, TGP._getindex(xs, t))
        ys[t] = yy
    end

    return (ys, state)
end



my_logpdf(ft::TGP.FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}}) =
    TGP.logpdf(TGP.build_lgssm(ft), y)
my_logpdf(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}}) =
    sum(TGP.scan_emit(TGP.step_logpdf, zip(model, y), model.transitions.x0, eachindex(model))[1])
function my_scan_emit(f, model, y, state)
    idx = eachindex(model)
    (lml, state) = f(state, model[1], y[1])
    lmls = Vector{typeof(yy)}(undef, length(idx))
    lmls[1] = lml
    for t in idx[2:end]
        (lml, state) = f(state, model[t], y[t])
        lmls[t] = lml
    end
    return lmls
end
function my_step_logpdf(x::TGP.Gaussian, (model_i, y_i))
    xp = my_predict(x, model_i.transition)
    xf, lml = my_posterior_and_lml(xp, model_i.emission, y_i)
    return lml, xf
end
my_predict(x::TGP.Gaussian, f::TGP.AbstractLGC) = TGP.Gaussian(f.A * x.m + f.a, f.A * my_symmetric(x.P) * f.A' + f.Q)
my_symmetric(X::AbstractMatrix) = Symmetric(X)
my_symmetric(X::Diagonal) = X
function my_posterior_and_lml(x::TGP.Gaussian, f::TGP.ScalarOutputLGC, y::T) where {T<:Real}
    m, P = x.m, x.P
    A, a, Q = f.A, f.a, f.Q
    V = A * P
    sqrtS = sqrt(V * A' + Q)  # std of obs at time
    B = sqrtS \ V
    α = sqrtS \ (y - (A * m + a))
    lml = -(log(2π) + 2 * log(sqrtS) + α^2) / 2
    return TGP.Gaussian(m + B'α, P - B'B), lml
end

## Looking into using a gutted version of TemporalGPs attempt 1
# LTI SDE : Linear Time Invariant Stochastic differential equation
# LGGSM : linear-Gaussian state-space model
# LGC : Linear-Gaussian Conditional
import TemporalGPs; TGP = TemporalGPs

fx = f2(x)
logpdf(fx, y)
@profiler for i in 1:100
    logpdf(fx, y)
end

# @enter logpdf(fx, y)
y_pr=y
# function AbstractGPs.logpdf(fx::FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})
x3, Σlmls, lmls, tr_indices, pr_indices = TGP.build_inference_data(
    fx.f, fx.x, fx.Σy, fill(missing, length(y_pr)),
)
Σlmls_pr = TGP.noise_var_to_time_form(fx.x, fx.Σy)
lmls_pr = TGP.observations_to_time_form(fx.x, y_pr)
Σlmls_pr_full = TGP.build_prediction_obs_vars(pr_indices, x3, Σlmls_pr)
lmls_pr_full = TGP.build_prediction_obs(tr_indices, pr_indices, x3, lmls_pr)
model = TGP.build_lgssm(fx.f.prior, x3, Σlmls)
model_post = TGP.replace_observation_noise_cov(posterior(model, lmls), Σlmls_pr_full)
logpdf(model_post, lmls_pr_full)

# @enter logpdf(model_post, lmls_pr_full)
model_with_missings, y_filled_in = TGP.transform_model_and_obs(model_post, lmls_pr_full)
logpdf(model_with_missings, y_filled_in) + TGP._logpdf_volume_compensation(lmls_pr_full, model_post)

# @enter _logpdf_volume_compensation
emissions = model_post.emissions
y_obs_count = sum(n -> lmls_pr_full[n] === missing ? TGP.dim_out(emissions[n]) : 0, eachindex(lmls_pr_full))
y_obs_count * log(2π * 1e15) / 2

# @enter logpdf(model_with_missings, y_filled_in)
f = TGP.step_logpdf; xs = zip(model_with_missings, y_filled_in); state = TGP.x0(model_with_missings); idx = eachindex(model_with_missings)
sum(TGP.scan_emit(f, xs, state, idx)[1])
TGP.scan_emit(f, xs, state, idx)[1]

# @enter function scan_emit(f, xs, state, idx)
(y, state) = f(state, TGP._getindex(xs, idx[1]))
lmls = Vector{typeof(y)}(undef, length(idx))
lmls[idx[1]] = y
for t in idx[2:end]
    (y, state) = f(state, TGP._getindex(xs, t))
    lmls[t] = y
end
lmls# , state

# @enter f (i.e. step_logpdf(x::Gaussian, (model, y)))
modelt, yt = TGP._getindex(xs, idx[1])
ordering = TGP.ordering(modelt)
TGP.step_logpdf(ordering, state, (modelt, yt))

# @enter function step_logpdf(::Reverse, x::Gaussian, (model, y))
ft = TGP.emission_dynamics(modelt)
_, lml = TGP.posterior_and_lml(state, ft, yt)

# @enter function posterior_and_lml(x::Gaussian, f::ScalarOutputLGC, y::T) where {T<:Real}
m, P = TGP.get_fields(state)
A, a, Q = TGP.get_fields(ft)
V = A * P
sqrtS = sqrt(V * A' + Q)
α = sqrtS \ (y2 - (A * m + a))
lml = -(log(2π) + 2 * log(sqrtS) + α^2) / 2

using Distributions

AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real}) = _logpdf(ft, y)

function AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return _logpdf(ft, y)
end

function _logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return logpdf(build_lgssm(ft), observations_to_time_form(ft.x, y))
end


function gp_reg(fx::TGP.FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})
    # function AbstractGPs.logpdf(fx::FinitePosteriorLTISDE, y_pr::AbstractVector{<:Real})
    x, Σlmls, lmls, tr_indices, pr_indices = TGP.build_inference_data(
        fx.f, fx.x, fx.Σy, fill(missing, length(y_pr)),
    )
    Σlmls_pr = TGP.noise_var_to_time_form(fx.x, fx.Σy)
    lmls_pr = TGP.observations_to_time_form(fx.x, y_pr)
    Σlmls_pr_full = TGP.build_prediction_obs_vars(pr_indices, x, Σlmls_pr)
    lmls_pr_full = TGP.build_prediction_obs(tr_indices, pr_indices, x, lmls_pr)
    model = TGP.build_lgssm(fx.f.prior, x, Σlmls)
    model_post = TGP.replace_observation_noise_cov(posterior(model, lmls), Σlmls_pr_full)

    model_with_missings, y_filled_in = TGP.transform_model_and_obs(model_post, lmls_pr_full)
    gp_reg(model_with_missings, y_filled_in)
end

function gp_reg(model::TGP.LGSSM, y::AbstractVector{<:Union{AbstractVector, <:Real}})
    f = step_logpdf; xs = zip(model, y); state = TGP.x0(model); idx = eachindex(model)
    sum(scan_emit(f, xs, state, idx))
end

function scan_emit(f, xs, state, idx)
    (y, state) = f(state, TGP._getindex(xs, idx[1]))
    lmls = Vector{typeof(y)}(undef, length(idx))
    lmls[idx[1]] = y
    for t in idx[2:end]
        (y, state) = f(state, TGP._getindex(xs, t))
        lmls[t] = y
    end
    lmls
end

function step_logpdf(x::Distributions.Gaussian, (modelt, yt))
    ordering = TGP.ordering(modelt)
    TGP.step_logpdf(ordering, state, (modelt, yt))
end

function step_logpdf(::TGP.Reverse, x::Distributions.Gaussian, (model, y))
    ft = TGP.emission_dynamics(modelt)
    return gp_reg_term(state, ft, yt)
end

function gp_reg_term(x::Distributions.Gaussian, f::TGP.ScalarOutputLGC, y::T) where {T<:Real}
    m, P = TGP.get_fields(state)
    A, a, Q = TGP.get_fields(ft)
    V = A * P
    sqrtS = sqrt(V * A' + Q)
    α = sqrtS \ (y2 - (A * m + a))
    # lml = -(log(2π) + 2 * log(sqrtS) + α^2) / 2
    return α*α
end


gp_reg(fx, y)

typeof(fx) <: TGP.FinitePosteriorLTISDE
## F all this noise

f_post = posterior(fix, y)
fx = f_post(x)
# # Compute the posterior marginals.
y_post = marginals(fx)



# Sample from the prior as usual.
# y = rand(fx)

# Compute the log marginal likelihood of the data as usual.
using BenchmarkTools
@btime -logpdf(fx, y)
-logpdf(SSOF.SOAP_gp(model.tel.log_λ), model.tel.lm.μ)
Σ = cov(fx)

kernel = PiecewisePolynomialKernel(;degree=3, dim=1)
kernel = Matern52Kernel()
f_naive = GP(kernel)
fx_naive = f_naive(x)

f = to_sde(f_naive, SArrayStorage(Float64))
fx = f(x)

@time cov(fx)
@time cov(fx_naive)
cov(fx_naive)

-logpdf(SSOF.SOAP_gp(model.tel.log_λ), model.tel.lm.μ)
-logpdf(fx, zeros(length(y)))
SSOF._loss
@time logpdf(fx, y)

model.rv.lm.s .= 0
model.tel.lm.s .= 0
model.rv.lm.s .= 0

function Distributions.logpdf(f::FiniteGP, Y::AbstractMatrix{<:Real})
    m, C_mat = mean_and_cov(f)
    C = cholesky(_symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(log(2π)) + logdet(C)) .+ diag_Xt_invA_X(C, Y .- m)) ./ 2
end



sum((data.flux .^ 2) ./ data.var)
