## Setup
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using Plots

stars = ["10700", "26965"]
star = stars[2]
plot_stuff = true
plot_stuff_fit = true
use_telstar = true
expres_data_path = "E:/telfitting/"
desired_order = 47  # 68 has a bunch of tels, 47 has very few
## Setting up necessary variables and functions

@load expres_data_path * star * "_$(desired_order).jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses

tf_output = SSOF.TFOutput(tf_model)

if use_telstar
    tf_workspace, loss = SSOF.TFWorkspaceTelStar(tf_model, tf_output, tf_data; return_loss_f=true)
else
    tf_workspace, loss = SSOF.TFWorkspace(tf_model, tf_output, tf_data; return_loss_f=true)
end

if plot_stuff_fit
    include("../src/_plot_functions.jl")
    status_plot(tf_output, tf_data)
end

light_speed_nu = 299792458
rvs_notel = (tf_model.rv.lm.s .* light_speed_nu)'
resid_stds = [std(rvs_notel)]
losses = [loss()]
tracker = 0
println("guess $tracker, std=$(round(std(rvs_notel), digits=5))")
rvs_notel_opt = copy(rvs_notel)

@time if !tf_model.todo[:reg_improved]
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    SSOF.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
    tf_model.todo[:reg_improved] = true
    tf_model.todo[:optimized] = false
    @save expres_data_path * star * "_$(desired_order).jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
end

if !tf_model.todo[:optimized]
    @time for i in 1:8
        SSOF.train_TFOrderModel!(tf_workspace)
        rvs_notel_opt[:] = (tf_model.rv.lm.s .* light_speed_nu)'

        append!(resid_stds, [std(rvs_notel_opt)])
        append!(losses, [loss()])
        if plot_stuff_fit; status_plot(tf_output, tf_data) end
        tracker += 1
        println("guess $tracker")
        println("loss   = $(losses[end])")
        println("rv std = $(round(std(rvs_notel_opt), digits=5))")
    end
    tf_model.todo[:optimized] = true
    @save expres_data_path * star * "_$(desired_order).jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
end

plot(0:tracker, resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(0:tracker, losses; xlabel="iter", ylabel="loss", legend=false)

plot_stellar_model_bases(tf_model)
plot_telluric_model_bases(tf_model)

## Plots

if plot_stuff
    include("../src/_plot_functions.jl")
    fig_dir = "EXPRES/figs/" * star * "/$(desired_order)/"
    mkpath(fig_dir)

    using CSV, DataFrames
    expres_output = CSV.read(expres_data_path * star* "_activity.csv", DataFrame)
    eo_rv = expres_output."CBC RV [m/s]"
    eo_rv_σ = expres_output."CBC RV Err. [m/s]"
    eo_time = expres_output."Time [MJD]"

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = (tf_model.rv.lm.s .* light_speed_nu)'
    predict_plot = plot_model_rvs(times_nu, rvs_naive, rvs_notel, rvs_notel_opt, eo_time, eo_rv, eo_rv_σ)
    png(predict_plot, fig_dir * "model_rvs.png")

    predict_plot = plot_stellar_model_bases(tf_model)
    png(predict_plot, fig_dir * "model_star_basis.png")

    predict_plot = plot_stellar_model_scores(tf_model)
    png(predict_plot, fig_dir * "model_star_weights.png")

    predict_plot = plot_telluric_model_bases(tf_model)
    png(predict_plot, fig_dir * "model_tel_basis.png")

    predict_plot = plot_telluric_model_scores(tf_model)
    png(predict_plot, fig_dir * "model_tel_weights.png")

    predict_plot = plot_stellar_model_bases(tf_model; inds=1:3)
    png(predict_plot, fig_dir * "model_star_basis_few.png")

    predict_plot = plot_stellar_model_scores(tf_model; inds=1:3)
    png(predict_plot, fig_dir * "model_star_weights_few.png")

    predict_plot = plot_telluric_model_bases(tf_model; inds=1:3)
    png(predict_plot, fig_dir * "model_tel_basis_few.png")

    predict_plot = plot_telluric_model_scores(tf_model; inds=1:3)
    png(predict_plot, fig_dir * "model_tel_weights_few.png")

    predict_plot = status_plot(tf_output, tf_data)
    png(predict_plot, fig_dir * "status_plot")
end
## TODO ERES presentation plots

hmm = status_plot(tf_output, tf_data)
png(hmm, "status_plot")
plot_stellar_model_bases(tf_model; inds=1:3)
hmm = plot_telluric_model_bases(tf_model; inds=1:3)
png(hmm, "telluric_plot")
anim = @animate for i in 1:40
    plt = plot_spectrum(; title="Telluric Spectrum")
    plot!(plt, exp.(tf_data.log_λ_obs[:, i]), view(tf_output.tel, :, i), label="", yaxis=[0.95, 1.005])
end
gif(anim, "show_telluric_var.gif", fps = 10)
