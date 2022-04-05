## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using Statistics
using JLD2

## Setting up necessary variables

stars = ["10700", "2021/12/10", "2021/12/19", "2021/12/20", "2021/12/23"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 81)  # 81 has a bunch of tels, 60 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 1)
recalc = SSOF.parse_args(4, Bool, false)
opt = SSOFU.valid_optimizers[which_opt]

## Loading in data and initializing model
base_path = neid_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
save_path = base_path * "results.jld2"

model, data, times_nu, airmasses = SSOFU.create_model(data_path, desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path, recalc=recalc)
mws = SSOFU.create_workspace(model, data, opt)
SSOFU.improve_regularization!(mws; save_fn=save_path)
SSOFU.improve_model!(mws; show_plot=interactive, save_fn=save_path)
mws, _, _, _, _, _ = SSOFU.downsize_model(mws, times_nu; save_fn=save_path, decision_fn=base_path*"model_decision.jld2", plots_fn=base_path)
# SSOFU.improve_regularization!(mws; save_fn=save_path, redo=true)
rvs, rv_errors = SSOFU.estimate_errors(mws; save_fn=save_path)

## Plots
@load neid_save_path * star * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv ord_has_rvs

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu, rvs, rv_errors, neid_time, neid_rv, neid_rv_σ; display_plt=interactive, markerstrokewidth=1, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
png(plt, save_path * "model_rvs.png")

SSOFU.save_model_plots(mws, airmasses, times_nu, base_path; display_plt=interactive)

if ord_has_rvs[desired_order]
    plt = SSOFU.plot_model_rvs(times_nu, rvs, rv_errors, neid_time, neid_order_rv[:, desired_order], zeros(length(times_nu)); display_plt=interactive, markerstrokewidth=1, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
    png(plt, save_path * "model_rvs_order.png")
end
