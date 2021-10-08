## Importing packages
using Pkg
Pkg.activate("EXPRES")

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using Plots
import StatsBase

## Setting up necessary variables

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 2)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 68)  # 68 has a bunch of tels, 47 has very few
use_reg = SSOF.parse_args(3, Bool, true)

## Loading in data and initializing model
save_path = expres_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end

if isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    end
else
    model_upscale = sqrt(2)
    # model_upscale = 2 * sqrt(2)
    @time model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8, upscale=model_upscale)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
    if !use_reg
        SSOF.zero_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Creating optimization workspace
workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(workspace.o, workspace.d)
else
    ENV["GKSwstype"] = "100"  # setting the GR workstation type to 100/nul
end

## Improving regularization

if !model.metadata[:todo][:reg_improved]
    @time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true, ignore_regularization=true)  # 16s
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(StatsBase.sample(1:n_obs, n_obs_train; replace=false))
    @time SSOF.fit_regularization!(model, data, training_inds)
    model.metadata[:todo][:reg_improved] = true
    model.metadata[:todo][:optimized] = false
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Optimizing model

if !model.metadata[:todo][:optimized]
    @time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
    rvs_notel_opt = SSOF.rvs(model)
    if interactive; status_plot(workspace.o, workspace.d) end
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end
status_plot(workspace.o, workspace.d)

## Downsizing model

if !model.metadata[:todo][:downsized]
    test_n_comp_tel = 0:8
    test_n_comp_star = 0:8
    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
    for (i, n_tel) in enumerate(test_n_comp_tel)
        for (j, n_star) in enumerate(test_n_comp_star)
            comp_ls[i, j], ks[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], model, data)
        end
    end
    n_comps_best, ℓ, aic, bic = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, data.var; return_inters=true)
    @save save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star

    model_large = copy(model)
    model = SSOF.downsize(model, n_comps_best[1], n_comps_best[2])
    model.metadata[:todo][:downsized] = true
    model.metadata[:todo][:reg_improved] = true
    workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)
    SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel # model_large
end


## Getting RV error bars (only regularization held constant)

if !model.metadata[:todo][:err_estimated]
    data.var[data.var.==Inf] .= 0
    data_noise = sqrt.(data.var)
    data.var[data.var.==0] .= Inf

    data_holder = copy(data)
    model_holder = copy(model)
    n = 50
    rv_holder = zeros(n, length(model.rv.lm.s))
    @time for i in 1:n
        data_holder.flux .= data.flux .+ (data_noise .* randn(size(data_holder.var)))
        SSOF.train_OrderModel!(SSOF.OptimWorkspace(model_holder, data_holder), f_tol=1e-8)
        rv_holder[i, :] = SSOF.rvs(model_holder)
    end
    rv_errors = vec(std(rv_holder; dims=1))
    model.metadata[:todo][:err_estimated] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel rv_errors
end

## Plots

if save_plots

    include(SSOF_path * "/src/_plot_functions.jl")

    using CSV, DataFrames
    expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
    eo_rv = expres_output."CBC RV [m/s]"
    eo_rv_σ = expres_output."CBC RV Err. [m/s]"
    eo_time = expres_output."Time [MJD]"

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = SSOF.rvs(model)
    plt = plot_model_rvs_new(times_nu, rvs_notel_opt, vec(rv_errors), eo_time, eo_rv, eo_rv_σ; display_plt=interactive, markerstrokewidth=1);
    png(plt, save_path * "model_rvs.png")

    if !(typeof(model.star.lm) <: SSOF.TemplateModel)
        plt = plot_stellar_model_bases(model; display_plt=interactive);
        png(plt, save_path * "model_star_basis.png")

        plt = plot_stellar_model_scores(model; display_plt=interactive);
        png(plt, save_path * "model_star_weights.png")
    end

    if !(typeof(model.tel.lm) <: SSOF.TemplateModel)
        plt = plot_telluric_model_bases(model; display_plt=interactive);
        png(plt, save_path * "model_tel_basis.png")

        plt = plot_telluric_model_scores(model; display_plt=interactive);
        png(plt, save_path * "model_tel_weights.png")
    end

    plt = status_plot(workspace.o, workspace.d; display_plt=interactive);
    png(plt, save_path * "status_plot.png")

    plt = component_test_plot(ℓ, test_n_comp_tel, test_n_comp_star);
    png(plt, save_path * "l_plot.png")

    plt = component_test_plot(aic, test_n_comp_tel, test_n_comp_star; ylabel="AIC");
    png(plt, save_path * "aic_plot.png")

    plt = component_test_plot(bic, test_n_comp_tel, test_n_comp_star; ylabel="BIC");
    png(plt, save_path * "bic_plot.png")
end
# comment
