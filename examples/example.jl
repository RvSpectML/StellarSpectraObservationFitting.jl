## Setup
using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

plot_stuff = true
use_telstar = true
improve_regularization = false

## Setting up necessary variables and functions

@load "E:/telfitting/model_150k.jld2" model n_obs data rvs_notel rvs_naive

if improve_regularization
    using StatsBase
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(sample(1:n_obs, n_obs_train; replace=false))
    SSOF.fit_regularization!(model, data, training_inds; use_telstar=use_telstar)
else
    model.reg_tel[:L2_μ] = 1e7
    model.reg_tel[:L1_μ] = 1e3
    model.reg_tel[:L1_μ₊_factor] = 2
    model.reg_tel[:L2_M] = 1e7
    model.reg_tel[:L1_M] = 1e5
    delete!(model.reg_tel, :shared_M)

    model.reg_star[:L2_μ] = 1e6
    model.reg_star[:L1_μ] = 1e-1
    model.reg_star[:L1_μ₊_factor] = 2
    model.reg_star[:L2_M] = 1e11
    model.reg_star[:L1_M] = 1e7
    delete!(model.reg_star, :shared_M)
end

output = SSOF.Output(model)

if use_telstar
    workspace, loss = SSOF.WorkspaceTelStar(model, output, data; return_loss_f=true)
else
    workspace, loss = SSOF.Workspace(model, output, data; return_loss_f=true)
end

using Plots
if plot_stuff
    @load "E:/telfitting/telfitting_workspace_smol_150k.jld2" airmasses planet_P_nu rvs_activ_no_noise rvs_activ_noisy rvs_kep_nu times_nu plot_times plot_rvs_kep true_tels

    plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (nm)", ylabel = "Continuum Normalized Flux", dpi = 400, kwargs...)
    plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = 400, kwargs...)

    function status_plot(o::SSOF.Output, d::SSOF.Data; plot_epoch::Int=10, tracker::Int=0)
        obs_λ = exp.(d.log_λ_obs[:, plot_epoch])
        l = @layout [a; b]
        # plt = plot_spectrum(; legend = :bottomleft, size=(800,1200), layout = l)
        # plt = plot_spectrum(; xlim=(627.8,628.3), legend=:bottomleft, size=(800,1200), layout = l) # o2
        # plt = plot_spectrum(; xlim=(651.5,652), legend=:bottomleft, size=(800,1200), layout = l)  # h2o
        plt = plot_spectrum(; xlim = (647, 656), legend = :bottomleft, size=(800,1200), layout = l)  # h2o
        plot!(plt[1], obs_λ, true_tels[:, plot_epoch], label="true tel")
        plot!(plt[1], obs_λ, d.flux[:, plot_epoch] ./ (o.star[:, plot_epoch] + o.rv[:, plot_epoch]), label="predicted tel", alpha = 0.5)
        plot!(plt[1], obs_λ, o.tel[:, plot_epoch], label="model tel: $tracker", alpha = 0.5)
        plot_star_λs = exp.(d.log_λ_star[:, plot_epoch])
        plot!(plt[2], plot_star_λs, d.flux[:, plot_epoch] ./ true_tels[:, plot_epoch], label="true star")
        plot!(plt[2], plot_star_λs, d.flux[:, plot_epoch] ./ o.tel[:, plot_epoch], label="predicted star", alpha = 0.5)
        plot!(plt[2], plot_star_λs, o.star[:, plot_epoch] + o.rv[:, plot_epoch], label="model star: $tracker", alpha = 0.5)
        display(plt)
    end
    status_plot(output, data)
end

rvs_notel = SSOF.rvs(model)
rvs_std(rvs; inds=:) = std((rvs - rvs_kep_nu[inds]) - rvs_activ_no_noise[inds])
resid_stds = [rvs_std(rvs_notel)]
losses = [loss()]
tracker = 0
println("guess $tracker, std=$(round(rvs_std(rvs_notel), digits=5))")
rvs_notel_opt = copy(rvs_notel)

@time for i in 1:8
    SSOF.train_OrderModel!(workspace)
    rvs_notel_opt[:] = SSOF.rvs(model)

    append!(resid_stds, [rvs_std(rvs_notel_opt)])
    append!(losses, [loss()])

    status_plot(output, data)
    tracker += 1
    println("guess $tracker")
    println("loss   = $(losses[end])")
    println("rv std = $(round(rvs_std(rvs_notel_opt), digits=5))")
end

plot(0:tracker, resid_stds; xlabel="iter", ylabel="predicted RV - active RV RMS", legend=false)
plot(0:tracker, losses; xlabel="iter", ylabel="loss", legend=false)

## Plots

using LinearAlgebra

if plot_stuff

    fig_dir = "examples/figs/"

    # Compare first guess at RVs to true signal
    plt = plot_rv()
    plot!(plt, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(plt, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    png(plt, fig_dir * "model_0_phase.png")

    # Compare RV differences to actual RVs from activity
    plt = plot_rv()
    plot!(plt, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(plt, times_nu, rvs_naive - rvs_kep_nu, st=:scatter, ms=3, color=:blue, label="Before model")
    png(plt, fig_dir * "model_0.png")

    # Compare second guess at RVs to true signal
    plt = plot_rv()
    plot!(plt, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(plt, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(plt, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(plt, fig_dir * "model_1_phase.png")

    # Compare RV differences to actual RVs from activity
    plt = plot_rv()
    plot!(plt, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(plt, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    png(plt, fig_dir * "model_1.png")

    # Compare second guess at RVs to true signal
    plt = plot_rv()
    plot!(plt, plot_times .% planet_P_nu, plot_rvs_kep, st=:line, color=:red, lw=1, label="Injected Keplerian")
    plot!(plt, times_nu .% planet_P_nu, rvs_naive, st=:scatter, ms=3, color=:blue, label="Before model")
    plot!(plt, times_nu .% planet_P_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(plt, times_nu .% planet_P_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(plt, fig_dir * "model_2_phase.png")

    # Compare RV differences to actual RVs from activity
    plt = plot_rv()
    plot!(plt, times_nu, rvs_activ_noisy, st=:scatter, ms=3, color=:red, label="Activity (with obs. SNR and resolution)")
    plot!(plt, times_nu, rvs_notel - rvs_kep_nu, st=:scatter, ms=3, color=:lightgreen, label="Before optimization")
    plot!(plt, times_nu, rvs_notel_opt - rvs_kep_nu, st=:scatter, ms=3, color=:darkgreen, label="After optimization")
    png(plt, fig_dir * "model_2.png")

    # plt = plot_spectrum(; xlim=(627.8,628.3)) # o2
    # plt = plot_spectrum(; xlim = (647, 656))  # h2o
    # plt = plot_spectrum(; xlim=(651.5,652))  # h2o
    plt = plot_spectrum(; title="Stellar model")
    plot!(model.star.λ, model.star.lm.M[:, 1] ./ norm(model.star.lm.M[:, 1]); label="basis 1")
    plot!(model.star.λ, model.star.lm.M[:, 2] ./ norm(model.star.lm.M[:, 2]); label="basis 2")
    plot!(model.star.λ, model.star.lm.μ; label="μ")
    png(plt, fig_dir * "model_star_basis.png")

    plt = plot_spectrum(; xlim=(647, 656), title="Stellar model")  # h2o
    plot!(model.star.λ, model.star.lm.M[:, 1] ./ norm(model.star.lm.M[:, 1]); label="basis 1")
    plot!(model.star.λ, model.star.lm.M[:, 2] ./ norm(model.star.lm.M[:, 2]); label="basis 2")
    plot!(model.star.λ, model.star.lm.μ; label="μ star")
    plot!(model.tel.λ, model.tel.lm.μ; label="μ tel")
    png(plt, fig_dir * "model_star_basis_h2o.png")

    plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = 400, kwargs...)
    plt = plot_scores(; title="Stellar model")
    scatter!(times_nu, model.star.lm.s[1, :] .* norm(model.star.lm.M[:, 1]); label="weights 1")
    scatter!(times_nu, model.star.lm.s[2, :] .* norm(model.star.lm.M[:, 2]); label="weights 2")
    png(plt, fig_dir * "model_star_weights.png")

    plt = plot_spectrum(; title="Telluric model")
    plot!(model.tel.λ, model.tel.lm.M[:, 1] ./ norm(model.tel.lm.M[:, 1]); label="basis 1")
    plot!(model.tel.λ, model.tel.lm.M[:, 2] ./ norm(model.tel.lm.M[:, 2]); label="basis 2")
    plot!(model.tel.λ, model.tel.lm.μ; label="μ")
    png(plt, fig_dir * "model_tel_basis.png")

    plt = plot_spectrum(;xlim=(647, 656), title="Telluric model (H2O)")  # h2o
    plot!(model.tel.λ, model.tel.lm.M[:,1] ./ norm(model.tel.lm.M[:, 1]); label="basis 1")
    plot!(model.tel.λ, model.tel.lm.M[:,2] ./ norm(model.tel.lm.M[:, 2]); label="basis 2")
    plot!(model.tel.λ, model.tel.lm.μ; label="μ")
    png(plt, fig_dir * "model_tel_basis_h2o.png")
    plt = plot_spectrum(;xlim=(627.8,628.3), title="Telluric model (O2)")  # o2
    plot!(model.tel.λ, model.tel.lm.M[:,1] ./ norm(model.tel.lm.M[:, 1]); label="basis 1")
    plot!(model.tel.λ, model.tel.lm.M[:,2] ./ norm(model.tel.lm.M[:, 2]); label="basis 2")
    plot!(model.tel.λ, model.tel.lm.μ; label="μ")
    png(plt, fig_dir * "model_tel_basis_o2.png")

    plt = plot_scores(; title="Telluric model")
    scatter!(times_nu, model.tel.lm.s[1, :] .* norm(model.tel.lm.M[:, 1]); label="weights 1")
    scatter!(times_nu, model.tel.lm.s[2, :] .* norm(model.tel.lm.M[:, 2]); label="weights 2")
    scatter!(times_nu, airmasses; label="airmasses")
    png(plt, fig_dir * "model_tel_weights.png")
end

std((rvs_naive - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_no_noise)
std((rvs_naive - rvs_kep_nu) - rvs_activ_noisy)
std((rvs_notel - rvs_kep_nu) - rvs_activ_noisy)
std((rvs_notel_opt - rvs_kep_nu) - rvs_activ_noisy)
std(rvs_activ_noisy - rvs_activ_no_noise)  # best case
std(rvs_notel_opt - rvs_kep_nu)
std(rvs_activ_noisy)
std(rvs_activ_no_noise)
