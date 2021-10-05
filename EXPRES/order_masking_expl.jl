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
include("data_locs.jl")  # defines expres_data_path and expres_save_path
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")

desired_order = 38

## Loading in data and initializing model
begin
    save_path = expres_save_path * star * "/$(desired_order)/"
    @load save_path * "data.jld2" n_obs data times_nu airmasses
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    end
    o = SSOF.Output(model, data)
    v = copy(data.var)
    v[v .== Inf] .= 3 * maximum(v[v .!= Inf])
    plot_epoch = argmin(mean(v; dims = 1))[2]
    sp(; plot_epoch=plot_epoch, kwargs...) = status_plot(o, data; plot_epoch=plot_epoch, kwargs...)
end
plot_epoch
sp(;plot_epoch=)

plot_telluric_model_bases(model)
plot_telluric_model_scores(model)

plot_stellar_model_bases(model)
plot_stellar_model_scores(model)

lims = [4190, 4202.6]
plot_telluric_model_bases(model); vline!(lims)
# plot_telluric_model_bases(model; xlim=lims)
plot_stellar_model_bases(model); vline!(lims)
# plot_stellar_model_bases(model; xlim=lims)
plt = sp(); vline!(plt[2], lims; label=""); vline!(plt[1], lims; label="")
# sp(; xlim=lims)

heatmap(exp.(data.log_λ_obs[:, 1]), 1:n_obs, data.var')
SSOF.mask_tellurics!()
SSOF.mask_tellurics!(data, log(4030), log(4035))
SSOF.mask_tellurics!(data, log(4076.9), log(4081))
