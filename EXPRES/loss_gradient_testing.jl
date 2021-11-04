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

## Loading in data and initializing model
save_path = expres_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end

## Making a model

# 7020, 114
ind_λ = 1:7020; ind_t = 1:114
data_small = SSOF.LSFData(data.flux[ind_λ, ind_t], data.var[ind_λ, ind_t], data.log_λ_obs[ind_λ, ind_t], data.log_λ_star[ind_λ, ind_t], data.lsf[ind_λ,ind_λ])

if false#isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    end
else
    model_upscale = sqrt(2)
    # model_upscale = 2 * sqrt(2)
    @time model = SSOF.OrderModel(data_small, "EXPRES", desired_order, star; n_comp_tel=3, n_comp_star=3, upscale=model_upscale)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data_small; use_gp=true)
    if !use_reg
        SSOF.zero_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    # @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Creating optimization workspace
workspace = SSOF.OptimWorkspace(model, data_small)

ts = workspace.telstar
@btime ts.obj.df(gn, ts.p0);

om = model; d = data_small; o = SSOF.Output(om, d); only_s=false
loss, loss_telstar, loss_telstar_s, loss_rv = SSOF.loss_funcs_telstar(o, om, d)

using Nabla
y = ts.unflatten(ts.p0)
loss_telstar(y)
gf = ∇(loss_telstar)
gf(y)


ts.obj.f(ts.p0)
ts.obj.df(ones(length(ts.p0)), ts.p0)

f = loss_telstar ∘ ts.unflatten
f(ts.p0)

θunfl = ts.unflatten(ts.p0)
using Nabla
∇(loss_telstar)(θunfl)
