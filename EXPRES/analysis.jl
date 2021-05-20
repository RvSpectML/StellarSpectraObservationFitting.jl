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

## Setting up necessary variables and functions

@load expres_data_path * star * ".jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses

tf_output = SSOF.TFOutput(tf_model)

if use_telstar
    tf_workspace, loss = SSOF.TFWorkspaceTelStar(tf_model, tf_output, tf_data; return_loss_f=true)
else
    tf_workspace, loss = SSOF.TFWorkspace(tf_model, tf_output, tf_data; return_loss_f=true)
end

if plot_stuff_fit
    plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
    plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

    function status_plot(tfo::SSOF.TFOutput, tfd::SSOF.TFData; plot_epoch::Int=10, tracker::Int=0)
        obs_λ = exp.(tfd.log_λ_obs[:, plot_epoch])
        l = @layout [a; b]
        predict_plot = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
        plot!(predict_plot[1], obs_λ, tfd.flux[:, plot_epoch] ./ (tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch]), label="predicted tel", alpha = 0.5)
        plot!(predict_plot[1], obs_λ, tfo.tel[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
        plot_star_λs = exp.(tfd.log_λ_star[:, plot_epoch])
        plot!(predict_plot[2], plot_star_λs, tfd.flux[:, plot_epoch] ./ tfo.tel[:, plot_epoch], label="predicted star", alpha = 0.5)
        plot!(predict_plot[2], plot_star_λs, tfo.star[:, plot_epoch] + tfo.rv[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
        display(predict_plot)
    end
    status_plot(tf_output, tf_data)
end

light_speed_nu = 299792458
rvs_notel = (tf_model.rv.lm.s .* light_speed_nu)'
resid_stds = [std(rvs_notel)]
losses = [loss()]
tracker = 0
println("guess $tracker, std=$(round(std(rvs_notel), digits=5))")
rvs_notel_opt = copy(rvs_notel)

if !tf_model.todo[:reg_improved]
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    SSOF.fit_regularization!(tf_model, tf_data, training_inds; use_telstar=use_telstar)
    tf_model.todo[:reg_improved] = true
    @save expres_data_path * star * ".jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
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
    @save expres_data_path * star * ".jld2" tf_model n_obs tf_data rvs_naive rvs_notel times_nu airmasses
end

plot(0:tracker, resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(0:tracker, losses; xlabel="iter", ylabel="loss", legend=false)

plot_stellar_model_bases(tf_model)
plot_telluric_model_bases(tf_model)

## Plots

if plot_stuff
    include("../src/_plot_functions.jl")
    fig_dir = "EXPRES/figs/" * star * "_"

    # Compare RV differences to actual RVs from activity
    predict_plot = plot_model_rvs(times_nu, rvs_naive, rvs_notel, rvs_notel_opt)
    png(predict_plot, fig_dir * "model_rvs.png")

    predict_plot = plot_stellar_model_bases(tf_model)
    png(predict_plot, fig_dir * "model_star_basis.png")

    predict_plot = plot_stellar_model_scores(tf_model)
    png(predict_plot, fig_dir * "model_star_weights.png")

    predict_plot = plot_telluric_model_bases(tf_model)
    png(predict_plot, fig_dir * "model_tel_basis.png")

    predict_plot = plot_telluric_model_scores(tf_model)
    png(predict_plot, fig_dir * "model_tel_weights.png")
end
