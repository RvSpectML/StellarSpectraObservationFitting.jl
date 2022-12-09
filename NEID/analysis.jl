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
using DataFrames, CSV

## Setting up necessary variables

stars = ["10700", "26965", "22049", "3651", "95735", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
solar = star_choice > 5
interactive = length(ARGS) == 0
if !interactive; ENV["GKSwstype"] = "100" end
include("data_locs.jl")  # defines neid_data_path and neid_save_path
desired_order = SSOF.parse_args(2, Int, 19)  # 81 has a bunch of tels, 60 has very few
log_lm = SSOF.parse_args(3, Bool, true)
dpca = SSOF.parse_args(4, Bool, false)
use_lsf = SSOF.parse_args(5, Bool, true)
recalc = SSOF.parse_args(6, Bool, true)
n_comp_tel, n_comp_star, use_custom_n_comp, recalc =
	SSOFU.how_many_comps(SSOF.parse_args(7, String, ""), recalc, desired_order)
save_folder = SSOF.parse_args(8, String, "aic")
use_custom_n_comp = SSOF.parse_args(9, Bool, false) && use_custom_n_comp
use_reg = SSOF.parse_args(10, Bool, true)
which_opt = SSOF.parse_args(11, Int, 1)
opt = SSOFU.valid_optimizers[which_opt]

## Loading in data and initializing model
base_path = neid_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
log_lm ? base_path *= "log_" : base_path *= "lin_"
dpca ? base_path *= "dcp_" : base_path *= "vil_"
use_lsf ? base_path *= "lsf/" : base_path *= "nol/"
mkpath(base_path)
if save_folder != ""
	base_path *= save_folder * "/"
	mkpath(base_path)
end
save_path = base_path * "results.jld2"
pipeline_path = neid_save_path * star * "/neid_pipeline.jld2"

data, times_nu, airmasses = SSOFU.get_data(data_path; use_lsf=use_lsf)
times_nu .-= 2400000.5

model = SSOFU.calculate_initial_model(data, "NEID", desired_order, star, times_nu;
	n_comp_tel=n_comp_tel, n_comp_star=n_comp_star, save_fn=save_path, plots_fn=base_path,
	recalc=recalc, use_reg=use_reg, use_custom_n_comp=use_custom_n_comp,
	dpca=dpca, log_lm=log_lm, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
	# log_λ_gp_tel=1/110000,
	log_λ_gp_tel=1/SSOFU.neid_neid_temporal_gp_lsf_λ(desired_order),
	careful_first_step=true, speed_up=true)
if all(isone.(model.tel.lm.μ)) && !SSOF.is_time_variable(model.tel); opt = "frozen-tel" end
mws = SSOFU.create_workspace(model, data, opt)
mkpath(base_path*"noreg/")
df_act = SSOFU.neid_activity_indicators(pipeline_path, data)
if !mws.om.metadata[:todo][:reg_improved]
	SSOFU.neid_plots(mws, airmasses, times_nu, SSOF.rvs(mws.om), zeros(length(times_nu)), star, base_path*"noreg/", pipeline_path, desired_order;
		display_plt=interactive, df_act=df_act);
end

SSOFU.improve_regularization!(mws; save_fn=save_path, careful_first_step=true, speed_up=true)
if !mws.om.metadata[:todo][:err_estimated]; SSOFU.improve_model!(mws, airmasses, times_nu; show_plot=interactive, save_fn=save_path, iter=500, verbose=true, careful_first_step=true, speed_up=true) end
rvs, rv_errors, tel_errors, star_errors = SSOFU.estimate_σ_curvature(mws; save_fn=base_path * "results_curv.jld2", save_model_fn=save_path, recalc=recalc, multithread=false)
mws.om.metadata[:todo][:err_estimated] = false
rvs_b, rv_errors_b, tel_s_b, tel_errors_b, star_s_b, star_errors_b, rv_holder, tel_holder, star_holder = SSOFU.estimate_σ_bootstrap(mws; save_fn=base_path * "results_boot.jld2", save_model_fn=save_path, recalc_mean=true, recalc=recalc, return_holders=true)

## Plots
SSOFU.neid_plots(mws, airmasses, times_nu, rvs_b, rv_errors_b, star, base_path, pipeline_path, desired_order;
	display_plt=interactive, df_act=df_act, tel_errors=tel_errors_b, star_errors=star_errors_b);
