## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using Plots

## Setting up necessary variables

stars = ["10700", "26965"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines expres_data_path and expres_save_path
use_telstar = SSOF.parse_args(2, Bool, true)
desired_order = SSOF.parse_args(3, Int, 68)  # 68 has a bunch of tels, 47 has very few

## Loading in data and initializing model
save_path = expres_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs tf_data times_nu airmasses

if isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" tf_model rvs_naive rvs_notel
else
    model_res = 2 * sqrt(2) * 150000
    @time tf_model = SSOF.TFOrderModel(tf_data, model_res, model_res, "EXPRES", desired_order, star; n_comp_tel=20, n_comp_star=20)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(tf_model, tf_data; use_gp=true)
    tf_model = SSOF.downsize(tf_model, 10, 10)
end

## Creating optimization workspace
if use_telstar
    tf_workspace, loss = SSOF.TFWorkspaceTelStar(tf_model, tf_data; return_loss_f=true)
else
    tf_workspace, loss = SSOF.TFWorkspaceTotal(tf_model, tf_data; return_loss_f=true)
end

## Plotting
SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(tf_workspace.tfo, tf_data)
end

light_speed_nu = 299792458
rvs_notel = (tf_model.rv.lm.s .* light_speed_nu)'
rvs_notel_opt = copy(rvs_notel)

if !tf_model.metadata[:todo][:reg_improved]
    @time results_telstar, _ = SSOF.train_TFOrderModel!(tf_workspace; print_stuff=true, ignore_regularization=true)  # 16s
    @time results_telstar, _ = SSOF.train_TFOrderModel!(tf_workspace; print_stuff=true, ignore_regularization=true, g_tol=SSOF._g_tol_def/10*sqrt(length(tf_workspace.telstar.p0)), f_tol=1e-8)  # 50s
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    @time SSOF.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
    tf_model.metadata[:todo][:reg_improved] = true
    tf_model.metadata[:todo][:optimized] = false
    @save save_path*"results.jld2" tf_model rvs_naive rvs_notel
end

if !tf_model.metadata[:todo][:optimized]
    @time results_telstar, _ = SSOF.train_TFOrderModel!(tf_workspace; print_stuff=true)  # 16s
    @time results_telstar, _ = SSOF.train_TFOrderModel!(tf_workspace; print_stuff=true, g_tol=SSOF._g_tol_def/10*sqrt(length(tf_workspace.telstar.p0)), f_tol=1e-8)  # 50s
    rvs_notel_opt[:] = (tf_model.rv.lm.s .* light_speed_nu)'
    if interactive; status_plot(tf_workspace.tfo, tf_data) end
    tf_model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" tf_model rvs_naive rvs_notel
end


## Getting RV error bars (only regularization held constant)

tf_data.var[tf_data.var.==Inf] .= 0
tf_data_noise = sqrt.(tf_data.var)
tf_data.var[tf_data.var.==0] .= Inf

tf_data_holder = copy(tf_data)
tf_model_holder = copy(tf_model)
n = 20
rv_holder = zeros(n, length(tf_model.rv.lm.s))
@time for i in 1:n
    tf_data_holder.flux[:, :] = tf_data.flux + (tf_data_noise .* randn(size(tf_data_holder.var)))
    SSOF.train_TFOrderModel!(SSOF.TFWorkspaceTelStar(tf_model_holder, tf_data_holder), g_tol=SSOF._g_tol_def/1*sqrt(length(tf_workspace.telstar.p0)), f_tol=1e-8)
    rv_holder[i, :] = (tf_model_holder.rv.lm.s .* light_speed_nu)'
end
rv_errors = std(rv_holder; dims=1)
@save save_path*"results.jld2" tf_model rvs_naive rvs_notel rv_errors

## Plots

if save_plots

    include(SSOF_path * "/src/_plot_functions.jl")

    using CSV, DataFrames
    expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
    eo_rv = expres_output."CBC RV [m/s]"
    eo_rv_σ = expres_output."CBC RV Err. [m/s]"
    eo_time = expres_output."Time [MJD]"

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = (tf_model.rv.lm.s .* light_speed_nu)'
    predict_plot = plot_model_rvs_new(times_nu, rvs_notel_opt, rv_errors, eo_time, eo_rv, eo_rv_σ; display_plt=interactive)
    png(predict_plot, save_path * "model_rvs.png")

    predict_plot = plot_stellar_model_bases(tf_model; display_plt=interactive)
    png(predict_plot, save_path * "model_star_basis.png")

    predict_plot = plot_stellar_model_scores(tf_model; display_plt=interactive)
    png(predict_plot, save_path * "model_star_weights.png")

    predict_plot = plot_telluric_model_bases(tf_model; display_plt=interactive)
    png(predict_plot, save_path * "model_tel_basis.png")

    predict_plot = plot_telluric_model_scores(tf_model; display_plt=interactive)
    png(predict_plot, save_path * "model_tel_weights.png")

    predict_plot = plot_stellar_model_bases(tf_model; inds=1:4, display_plt=interactive)
    png(predict_plot, save_path * "model_star_basis_few.png")

    predict_plot = plot_stellar_model_scores(tf_model; inds=1:4, display_plt=interactive)
    png(predict_plot, save_path * "model_star_weights_few.png")

    predict_plot = plot_telluric_model_bases(tf_model; inds=1:4, display_plt=interactive)
    png(predict_plot, save_path * "model_tel_basis_few.png")

    predict_plot = plot_telluric_model_scores(tf_model; inds=1:4, display_plt=interactive)
    png(predict_plot, save_path * "model_tel_weights_few.png")

    predict_plot = status_plot(tf_workspace.tfo, tf_data; display_plt=interactive)
    png(predict_plot, save_path * "status_plot.png")
end
