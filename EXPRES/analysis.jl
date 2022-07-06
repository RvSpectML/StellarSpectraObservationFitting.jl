## Importing packages
using Pkg
Pkg.activate("EXPRES")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using Statistics
using JLD2
using Plots

## Setting up necessary variables

stars = ["10700", "26965", "34411"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 68)  # 68 has a bunch of tels, 47 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 1)
recalc = SSOF.parse_args(5, Bool, false)
dpca = SSOF.parse_args(6, Bool, false)
use_lsf = SSOF.parse_args(7, Bool, false)
opt = SSOFU.valid_optimizers[which_opt]

## Loading in data and initializing model
base_path = expres_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
if !dpca
	base_path *= "wobble/"
	mkpath(base_path)
end
save_path = base_path * "results.jld2"

model, data, times_nu, airmasses, lm_tel, lm_star = SSOFU.create_model(data_path, desired_order, "EXPRES", star; use_reg=use_reg, save_fn=save_path, recalc=recalc, dpca=dpca)
if !use_lsf; data = SSOF.GenericData(data) end
if all(isone.(model.tel.lm.μ)) && !SSOF.is_time_variable(model.tel); opt = "frozen-tel" end
mws = SSOFU.create_workspace(model, data, opt)
mws = SSOFU.downsize_model(mws, times_nu, lm_tel, lm_star; save_fn=save_path, decision_fn=base_path*"model_decision.jld2", plots_fn=base_path)
SSOFU.improve_regularization!(mws; save_fn=save_path)
SSOFU.improve_model!(mws, airmasses, times_nu; show_plot=interactive, save_fn=save_path, iter=300)
# SSOFU.improve_regularization!(mws; save_fn=save_path, redo=true)
rvs, rv_errors, tel_errors, star_errors = SSOFU.estimate_errors(mws; save_fn=save_path)

## Plots
using CSV, DataFrames
expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu, rvs, rv_errors, eo_time, eo_rv, eo_rv_σ; display_plt=interactive, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
png(plt, base_path * "model_rvs.png")

SSOFU.save_model_plots(mws, airmasses, times_nu, base_path; display_plt=interactive, tel_errors=tel_errors, star_errors=star_errors)
