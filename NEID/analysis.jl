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
desired_order = SSOF.parse_args(2, Int, 81)  # 81 has a bunch of tels, 60 has very few
log_lm = SSOF.parse_args(3, Bool, true)
dpca = SSOF.parse_args(4, Bool, false)
use_lsf = SSOF.parse_args(5, Bool, true)
recalc = SSOF.parse_args(6, Bool, false)
use_custom_n_comp = SSOF.parse_args(7, Bool, true)
use_reg = SSOF.parse_args(8, Bool, true)
which_opt = SSOF.parse_args(9, Int, 1)
opt = SSOFU.valid_optimizers[which_opt]

if use_custom_n_comp
	df_n_comp = DataFrame(CSV.File("NEID/n_comps.csv"))
	i_df = desired_order - 3
	@assert df_n_comp[i_df, :order] == desired_order
	use_custom_n_comp = df_n_comp[i_df, :redo]
end

if use_custom_n_comp
	recalc = recalc || use_custom_n_comp
	n_comp_tel = df_n_comp[i_df, :n_tel_by_eye]
	n_comp_star = df_n_comp[i_df, :n_star_by_eye]
	better_model = df_n_comp[i_df, :better_model]
	# remove_reciprocal_continuum = df[i, :has_reciprocal_continuum]
	remove_reciprocal_continuum = false
	pairwise = false
else
	n_comp_tel = 5
	n_comp_star = 5
	remove_reciprocal_continuum = false
	pairwise = true
end

## Loading in data and initializing model
base_path = neid_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
log_lm ? base_path *= "log_" : base_path *= "lin_"
dpca ? base_path *= "dcp_" : base_path *= "vil_"
use_lsf ? base_path *= "lsf/" : base_path *= "nol/"
mkpath(base_path)
save_path = base_path * "results.jld2"
init_path = base_path * "results_init.jld2"
pipeline_path = neid_save_path * star * "/neid_pipeline.jld2"

if solar
    @load neid_save_path * "10700/$(desired_order)/results.jld2" model
	seed = model
    model, data, times_nu, airmasses = SSOFU.create_model(data_path,
		desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path,
		recalc=recalc, seed=seed, dpca=dpca, log_lm=log_lm, n_comp_tel=n_comp_tel,
		n_comp_star=n_comp_star, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
		log_λ_gp_tel=1/SSOFU.neid_neid_temporal_gp_lsf_λ(desired_order))
else
    model, data, times_nu, airmasses = SSOFU.create_model(data_path,
		desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path,
		recalc=recalc, dpca=dpca, log_lm=log_lm, n_comp_tel=n_comp_tel,
		n_comp_star=n_comp_star, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
		log_λ_gp_tel=1/SSOFU.neid_neid_temporal_gp_lsf_λ(desired_order))
end
times_nu .-= 2400000.5
lm_tel, lm_star, stellar_dominated = SSOFU.initialize_model!(model, data; init_fn=init_path, recalc=recalc, remove_reciprocal_continuum=remove_reciprocal_continuum, pairwise=pairwise)
if all(isone.(model.tel.lm.μ)) && !SSOF.is_time_variable(model.tel); opt = "frozen-tel" end
if !use_lsf; data = SSOF.GenericData(data) end
mws = SSOFU.create_workspace(model, data, opt)
# mws = SSOFU._downsize_model(mws, [2,0], 1, lm_tel, lm_star; print_stuff=true, ignore_regularization=true)
if use_custom_n_comp
	println("using by-eye number of basis vectors")
	base_path *= "by_eye/"
	save_path = base_path * "results.jld2"
	mkpath(base_path)
	mws = SSOFU._downsize_model(mws, [n_comp_tel, n_comp_star], better_model, lm_tel, lm_star; print_stuff=true, ignore_regularization=true, lm_tel_ind=2, lm_star_ind=2)
else
	solar ? base_path *= "bic/" : base_path *= "aic/"
	save_path = base_path * "results.jld2"
	mkpath(base_path)
	mws = SSOFU.downsize_model(mws, times_nu, lm_tel, lm_star; save_fn=save_path, decision_fn=base_path*"model_decision.jld2", plots_fn=base_path, use_aic=!solar)
end
mkpath(base_path*"noreg/")
df_act = SSOFU.neid_activity_indicators(pipeline_path, data)
SSOFU.neid_plots(mws, airmasses, times_nu, SSOF.rvs(mws.om), zeros(length(times_nu)), star, base_path*"noreg/", pipeline_path, desired_order;
	display_plt=interactive, df_act=df_act);
SSOFU.improve_regularization!(mws; save_fn=save_path)
if !mws.om.metadata[:todo][:err_estimated]; SSOFU.improve_model!(mws, airmasses, times_nu; show_plot=interactive, save_fn=save_path, iter=500, print_stuff=true) end
rvs, rv_errors, tel_errors, star_errors = SSOFU.estimate_σ_curvature(mws; save_fn=base_path * "results_curv.jld2", recalc=recalc, multithread=false)
mws.om.metadata[:todo][:err_estimated] = false
rvs_b, rv_errors_b, tel_s_b, tel_errors_b, star_s_b, star_errors_b, rv_holder, tel_holder, star_holder = SSOFU.estimate_σ_bootstrap(mws; save_fn=base_path * "results_boot.jld2", recalc_mean=true, recalc=recalc, return_holders=true)

## Plots
SSOFU.neid_plots(mws, airmasses, times_nu, rvs_b, rv_errors_b, star, base_path, pipeline_path, desired_order;
	display_plt=interactive, df_act=df_act, tel_errors=tel_errors_b, star_errors=star_errors_b);
