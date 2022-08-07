## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using Statistics
using JLD2
using Plots

## Setting up necessary variables

stars = ["10700", "26965", "22049", "3651", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
solar = star_choice > 5
interactive = length(ARGS) == 0
if !interactive; ENV["GKSwstype"] = "100" end
include("data_locs.jl")  # defines neid_data_path and neid_save_path
desired_order = SSOF.parse_args(2, Int, 81)  # 81 has a bunch of tels, 60 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 1)
recalc = SSOF.parse_args(5, Bool, false)
dpca = SSOF.parse_args(6, Bool, false)
use_lsf = SSOF.parse_args(7, Bool, false)
opt = SSOFU.valid_optimizers[which_opt]


## Loading in data and initializing model
base_path = neid_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
if dpca
	base_path *= "dpca/"
	mkpath(base_path)
else
	base_path *= "wobble/"
	mkpath(base_path)
end
save_path = base_path * "results.jld2"
init_path = base_path * "results_init.jld2"

if solar
    @load neid_save_path * "10700/$(desired_order)/results.jld2" model
	seed = model
    model, data, times_nu, airmasses = SSOFU.create_model(data_path, desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path, recalc=recalc, seed=seed, dpca=dpca)
else
    model, data, times_nu, airmasses = SSOFU.create_model(data_path, desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path, recalc=recalc, dpca=dpca)
end
times_nu .-= 2400000.5
lm_tel, lm_star = SSOFU.initialize_model!(model, data; init_fn=init_path, recalc=recalc)
if all(isone.(model.tel.lm.μ)) && !SSOF.is_time_variable(model.tel); opt = "frozen-tel" end
if !use_lsf; data = SSOF.GenericData(data) end
mws = SSOFU.create_workspace(model, data, opt)
# mws = SSOFU._downsize_model(mws, [1,1], 1, lm_tel, lm_star; print_stuff=true, ignore_regularization=true)
mws = SSOFU.downsize_model(mws, times_nu, lm_tel, lm_star; save_fn=save_path, decision_fn=base_path*"model_decision.jld2", plots_fn=base_path, use_aic=!solar)
pipeline_path = neid_save_path * star * "/neid_pipeline.jld2"
mkpath(base_path*"noreg/")
SSOFU.neid_plots(mws, airmasses, times_nu, SSOF.rvs(mws.om), zeros(length(times_nu)), star, base_path*"noreg/", pipeline_path, desired_order;
	display_plt=interactive);
SSOFU.improve_regularization!(mws; save_fn=save_path)
SSOFU.improve_model!(mws, airmasses, times_nu; show_plot=interactive, save_fn=save_path, iter=300, print_stuff=true)
rvs, rv_errors, tel_errors, star_errors = SSOFU.estimate_errors(mws; save_fn=save_path)

## Plots
df_act = SSOFU.neid_activity_indicators(pipeline_path, data)
SSOFU.neid_plots(mws, airmasses, times_nu, rvs, rv_errors, star, base_path, pipeline_path, desired_order;
	display_plt=interactive, df_act=df_act, tel_errors=tel_errors, star_errors=star_errors);
