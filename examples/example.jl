using Pkg
Pkg.activate("examples")
Pkg.instantiate()

import StellarSpectraObservationFitting as SSOF

using JLD2  # importing saved model
using Plots  # plotting
using Statistics  # mean function

# approximating PDF of saved model scores
using KernelDensity  # Bandwidth calculation
using Distributions  # Normal distribution
using StatsBase  # sample function

include("_plots.jl")  # some plotting functions
_plot(; dpi = 100, size = (960, 540), thickness_scaling=1., margin=4Plots.mm, kwargs...) =
    plot(; dpi=dpi, size=size, thickness_scaling=thickness_scaling, margin=margin, kwargs...)

# load in a prefit SSOF model and the data that it was fit to
@load "examples/data/results.jld2" model
@load "examples/data/data.jld2" data  # only used for LSF and blaze function

# how many observations we want
n_simulated_observations = 50

# Adding whatever additional RVs we want
injected_rvs = zeros(n_simulated_observations)
# K = 10  # planet velocity semi-amplitude (m/s)
# P = 60  # planet period (d)
# injected_rvs = K .* sin(times ./ P * 2 * π)

# simulated central pixel SNR
desired_max_snr = 500;

# spread observations evenly across all of 2022 (JD)
times = 2459580 .+ collect(LinRange(365., 0, n_simulated_observations));

# # Barycorrpy version
# using PyCall
# barycorrpy = PyNULL()
# # pyimport_conda("scipy", "scipy")
# # pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
# copy!(barycorrpy , pyimport("barycorrpy") )
# barycentric_corrections = barycorrpy.get_BC_vel(JDUTC=times, hip_id=19849, obsname="KPNO", ephemeris="de430")
# barycentric_rvs = barycentric_corrections[1]

# Epicyclic approximation for simplicity (correct within 411 m/s)
year = 365.25
phases = times ./ year * 2 * π
design_matrix = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
epicyclic_barycentric_rv_coefficients = [-20906.74122340397, -15770.355782489662, -390.29975114321905, -198.97407208182858, -67.99370656806558]
barycentric_rvs = design_matrix * epicyclic_barycentric_rv_coefficients

# observed wavelengths will just be the same as the original model
# could downsample here
log_λ_obs = model.tel.log_λ * (ones(n_simulated_observations)')
log_λ_stellar = log_λ_obs .+ (SSOF.rv_to_D(barycentric_rvs)')

# getting stellar feature score values that roughly match the original modeled distribution
bandwidth_stellar = KernelDensity.default_bandwidth(vec(model.star.lm.s))
model_scores_stellar = rand.(Normal.(sample(vec(model.star.lm.s), n_simulated_observations; replace=true), bandwidth_stellar))'
model_scores_stellar .-= mean(model_scores_stellar)

# evaluating linear model
_flux_stellar = SSOF._eval_lm(model.star.lm.M, model_scores_stellar, model.star.lm.μ; log_lm=model.star.lm.log)  # replace s for custom data

# interpolating stellar flux
b2o = SSOF.StellarInterpolationHelper(model.star.log_λ, injected_rvs + barycentric_rvs, log_λ_obs)
flux_stellar = SSOF.spectra_interp(_flux_stellar, injected_rvs + barycentric_rvs, b2o)

# getting telluric feature score values that roughly match the original modeled distribution
bandwidth_telluric = KernelDensity.default_bandwidth(vec(model.tel.lm.s))
model_scores_telluric = rand.(Normal.(sample(vec(model.tel.lm.s), n_simulated_observations; replace=true), bandwidth_telluric))'
model_scores_telluric .-= mean(model_scores_telluric)

# simulated telluric transmission
flux_tellurics = SSOF._eval_lm(model.tel.lm.M, model_scores_telluric, model.tel.lm.μ; log_lm=model.tel.lm.log)

flux_total = SSOF.total_model(flux_tellurics, flux_stellar)

## getting a realistic LSF
include("_lsf.jl")  # defines NEIDLSF.NEID_lsf()
lsf_simulated = NEIDLSF.neid_lsf(81, vec(mean(data.log_λ_obs; dims=2)), vec(mean(log_λ_obs; dims=2)))
flux_total_lsf = lsf_simulated * flux_total


# measured_blaze = data.flux ./ data.var
# _, smooth_blazes = SSOF.calc_continuum(exp.(data.log_λ_obs), measured_blaze, data.var)
# smooth_blaze = vec(mean(smooth_blazes; dims=2))
# # filling in gaps
# lo = Inf
# for i in eachindex(smooth_blaze)
# 	if isfinite(smooth_blaze[i])
# 		if isfinite(lo)
# 			println("$lo, $i")
# 			smooth_blaze[lo:i] .= smooth_blaze[lo] .+ ((smooth_blaze[i]-smooth_blaze[lo]) * LinRange(0,1,i-lo+1))
# 			lo = Inf
# 		end
# 	elseif !isfinite(lo)
# 		lo = i-1
# 	end
# end
# using DataInterpolations
# blaze_function= DataInterpolations.LinearInterpolation(smooth_blaze, vec(mean(data.log_λ_obs; dims=2)))
# @save "examples/data/blaze.jld2" blaze_function
@load "examples/data/blaze.jld2" blaze_function
var_total = flux_total_lsf ./ blaze_function.(log_λ_obs)

# scaling variance to have the desired SNR in the center
cp = Int(round(size(flux_total_lsf,1)/2))
snr_cp = median(flux_total_lsf[(cp-500):(cp+500),:] ./ sqrt.(var_total[(cp-500):(cp+500),:]))
var_total .*= (snr_cp / desired_max_snr)^2

flux_noisy = flux_total_lsf .+ (randn(size(var_total)) .* sqrt.(var_total))

data_simulated = SSOF.LSFData(flux_noisy, var_total, copy(var_total), log_λ_obs, log_λ_stellar, lsf_simulated) 
SSOF.mask_bad_edges!(data_simulated);

# getting initial model
star = "26965"
instrument = "SSOF"
order = 81

n_comp = 2  # maximum amount of feature vectors to use for each portion of the model

model_new = SSOF.calculate_initial_model(data_simulated; 
	instrument=instrument, desired_order=order, star=star, times=times,
	max_n_tel=n_comp, max_n_star=n_comp, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
	log_λ_gp_tel=5.134684755671457e-6,
	tel_log_λ=model.tel.log_λ, star_log_λ=model.star.log_λ, oversamp=false)

# setting up model workspace
mws = SSOF.ModelWorkspace(model_new, data_simulated);

SSOF.fit_regularization!(mws)

results = SSOF.improve_model!(mws; iter=500, verbose=true, careful_first_step=true, speed_up=false)

rvs, rvs_σ, tel_s_σ, star_s_σ = SSOF.estimate_σ_curvature(mws)
# rvs, rvs_σ, tel_s_σ, star_s_σ = SSOF.estimate_σ_bootstrap(mws)