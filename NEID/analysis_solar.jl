## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using JLD2
using Statistics
import StatsBase

## Setting up necessary variables

dates = ["2021/12/10", "2021/12/19", "2021/12/20", "2021/12/23"]
date = dates[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines neid_data_path and neid_save_path
desired_order = SSOF.parse_args(2, Int, 67)
use_reg = SSOF.parse_args(3, Bool, true)
which_opt = SSOF.parse_args(4, Int, 1)
recalc = SSOF.parse_args(5, Bool, false)
oversamp = SSOF.parse_args(6, Bool, true)
use_lsf = SSOF.parse_args(7, Bool, false)
use_gp_prior = SSOF.parse_args(8, Bool, true)
max_components = 3

## Loading in data and initializing model
save_path = neid_save_path * date * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end
if which_opt != 1
    save_path *= "adam_"
end
if !use_gp_prior
    save_path *= "nogpprior_"
end
if !oversamp
    save_path *= "undersamp_"
end
if !use_lsf
    data = SSOF.GenericData(data)
    save_path *= "nolsf_"
end

# takes a couple mins now
if isfile(save_path*"results.jld2") && !recalc
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    # if model.metadata[:todo][:downsized]
    #     @load save_path*"model_decision.jld2" comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star
    # end
else
    model_upscale = 2 * sqrt(2)
    @time model = SSOF.OrderModel(data, "NEID", desired_order, date; n_comp_tel=max_components, n_comp_star=max_components, upscale=model_upscale, oversamp=oversamp)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
    if !use_reg
        SSOF.rm_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end
if use_gp_prior
    delete!(model.reg_tel, :L2_μ)
    delete!(model.reg_star, :L2_μ)
    # delete!(model.reg_tel, :L2_M)
    # delete!(model.reg_star, :L2_M)
    delete!(model.reg_tel, :GP_M)
    delete!(model.reg_star, :GP_M)
else
    delete!(model.reg_tel, :GP_μ)
    delete!(model.reg_star, :GP_μ)
    delete!(model.reg_tel, :GP_M)
    delete!(model.reg_star, :GP_M)
end

## Creating optimization workspace
if which_opt == 1
    mws = SSOF.OptimWorkspace(model, data)
elseif which_opt == 2
    mws = SSOF.TelStarWorkspace(model, data)
else
    mws = SSOF.TotalWorkspace(model, data)
end

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(mws)
else
    ENV["GKSwstype"] = "100"  # setting the GR workstation type to 100/nul
end

## Improving regularization

if !model.metadata[:todo][:reg_improved]  # 27 mins
    @time SSOF.train_OrderModel!(mws; print_stuff=true, ignore_regularization=true)  # 45s
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(StatsBase.sample(1:n_obs, n_obs_train; replace=false))
    @time SSOF.fit_regularization!(mws, training_inds)
    model.metadata[:todo][:reg_improved] = true
    model.metadata[:todo][:optimized] = false
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Optimizing model

if !model.metadata[:todo][:optimized]
    @time results = SSOF.fine_train_OrderModel!(mws; print_stuff=true)  # 120s
    rvs_notel_opt = SSOF.rvs(model)
    if interactive; status_plot(mws) end
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end


## Getting RV error bars (only regularization held constant)

@time if !model.metadata[:todo][:err_estimated] # 25 mins
    data.var[data.var.==Inf] .= 0
    data_noise = sqrt.(data.var)
    data.var[data.var.==0] .= Inf

    data_holder = copy(data)
    model_holder = copy(model)
    n = 50
    rv_holder = Array{Float64}(undef, n, length(model.rv.lm.s))
    @time for i in 1:n
        data_holder.flux .= data.flux .+ (data_noise .* randn(size(data_holder.var)))
        SSOF.train_OrderModel!(typeof(mws)(model_holder, data_holder))
        rv_holder[i, :] = SSOF.rvs(model_holder)
    end
    rv_errors = vec(std(rv_holder; dims=1))
    model.metadata[:todo][:err_estimated] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel rv_errors
end

## Plots

if save_plots

    include(SSOF_path * "/src/_plot_functions.jl")

    @load neid_save_path * date * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv ord_has_rvs

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = SSOF.rvs(model)
    plt = plot_model_rvs(times_nu, rvs_notel_opt, vec(rv_errors), neid_time, neid_rv, neid_rv_σ; display_plt=interactive, markerstrokewidth=1, title="$date (median σ: $(round(median(vec(rv_errors)), digits=3)))");
    png(plt, save_path * "model_rvs.png")

    if ord_has_rvs[desired_order]
        plt = plot_model_rvs(times_nu, rvs_notel_opt, vec(rv_errors), neid_time, neid_order_rv[:, desired_order], zeros(n_obs); display_plt=interactive, markerstrokewidth=1, title="$date (median σ: $(round(median(vec(rv_errors)), digits=3)))");
        png(plt, save_path * "model_rvs_order.png")
    end

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

    plt = status_plot(mws; display_plt=interactive);
    png(plt, save_path * "status_plot.png")

    plt = component_test_plot(ℓ, test_n_comp_tel, test_n_comp_star);
    png(plt, save_path * "l_plot.png")

    plt = component_test_plot(aics, test_n_comp_tel, test_n_comp_star; ylabel="AIC");
    png(plt, save_path * "aic_plot.png")

    plt = component_test_plot(bics, test_n_comp_tel, test_n_comp_star; ylabel="BIC");
    png(plt, save_path * "bic_plot.png")
end
