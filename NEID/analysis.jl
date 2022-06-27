## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using Statistics
using JLD2
using Plots

## Setting up necessary variables

stars = ["10700", "26965", "9407", "185144", "22049", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
solar = star_choice > 5
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines neid_data_path and neid_save_path
desired_order = SSOF.parse_args(2, Int, 81)  # 81 has a bunch of tels, 60 has very few
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 1)
recalc = SSOF.parse_args(5, Bool, false)
dpca = SSOF.parse_args(6, Bool, true)
opt = SSOFU.valid_optimizers[which_opt]


## Loading in data and initializing model
base_path = neid_save_path * star * "/$(desired_order)/"
data_path = base_path * "data.jld2"
if !dpca
	base_path *= "wobble/"
	mkpath(base_path)
end
save_path = base_path * "results.jld2"

if solar
    @load neid_save_path * "10700/$(desired_order)/results.jld2" model
	seed = model
    model, data, times_nu, airmasses, lm_tel, lm_star = SSOFU.create_model(data_path, desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path, recalc=recalc, seed=seed, dpca=dpca)
else
    model, data, times_nu, airmasses, lm_tel, lm_star = SSOFU.create_model(data_path, desired_order, "NEID", star; use_reg=use_reg, save_fn=save_path, recalc=recalc, dpca=dpca)
end
mws = SSOFU.create_workspace(model, data, opt)
mws = SSOFU.downsize_model(mws, times_nu, lm_tel, lm_star; save_fn=save_path, decision_fn=base_path*"model_decision.jld2", plots_fn=base_path, use_aic=!solar)
SSOFU.improve_regularization!(mws; save_fn=save_path)
SSOFU.improve_model!(mws, airmasses, times_nu; show_plot=interactive, save_fn=save_path, iter=300)
rvs, rv_errors, tel_errors, star_errors = SSOFU.estimate_errors(mws; save_fn=save_path)

## Plots
@load neid_save_path * star * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv d_act_tot neid_tel d_lcs

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu, rvs, rv_errors, neid_time, neid_rv, neid_rv_σ; display_plt=interactive, markerstrokewidth=1, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
png(plt, base_path * "model_rvs.png")

lo, hi = exp.(quantile(vec(data.log_λ_star), [0.05, 0.95]))
df_act = Dict()
for wv in keys(d_lcs)
	if lo < parse(Float64, wv) < hi
		for key in d_lcs[wv]
			df_act[key] = d_act_tot[key]
			df_act[key*"_σ"] = d_act_tot[key*"_σ"]
		end
	end
end
SSOFU.save_model_plots(mws, airmasses, times_nu, base_path; display_plt=interactive, tel_errors=tel_errors, star_errors=star_errors, df_act=df_act)

if all(.!iszero.(view(neid_order_rv, :, desired_order)))
    plt = SSOFU.plot_model_rvs(times_nu, rvs, rv_errors, neid_time, view(neid_order_rv, :, desired_order), zeros(length(times_nu)); display_plt=interactive, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
    png(plt, base_path * "model_rvs_order.png")
end
