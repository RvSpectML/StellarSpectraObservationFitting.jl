## Importing packages
using Pkg
Pkg.activate("EXPRES")

import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using JLD2
using Statistics
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

function reset_model(; overrule::Bool=false)
    if !overrule && isfile(save_path*"results.jld2")
        @load save_path*"results.jld2" model rvs_naive rvs_notel
        if model.metadata[:todo][:err_estimated]
            @load save_path*"results.jld2" rv_errors
        end
        if model.metadata[:todo][:downsized]
            @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
        end
    else
        # model_upscale = sqrt(2)
        model_upscale = 2 * sqrt(2)
        @time model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8, upscale=model_upscale)
        @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
        if !use_reg
            SSOF.zero_regularization(model)
            model.metadata[:todo][:reg_improved] = true
        end
        @save save_path*"results.jld2" model rvs_naive rvs_notel
    end
    return model
end

# SSOF_path = dirname(dirname(pathof(SSOF)))
# include(SSOF_path * "/src/_plot_functions.jl")
# status_plot(workspace.o, workspace.d)
