## Importing packages
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

## Simulating data
@load "examples/jld2/results.jld2"

# how many observations we want(spread evenly across all of 2022 (JD))
n_simulated_observations = 50
times = 2459580 .+ collect(LinRange(365., 0, n_simulated_observations))  

# observed wavelengths will just be the same as the original model
# downsample here
log_λ_obs = model.tel.log_λ * ones(n_simulated_observations)'

# getting telluric feature score values that roughly match the original modeled distribution
bandwidth_telluric = KernelDensity.default_bandwidth(vec(model.tel.lm.s))
model_scores_telluric = rand.(Normal.(sample(vec(model.tel.lm.s), n_simulated_observations; replace=true), bandwidth_telluric))'

# simulated telluric transmission
flux_tellurics = SSOF._eval_lm(model.tel.lm.M, model_scores_telluric, model.tel.lm.μ; log_lm=model.tel.lm.log)  # replace s for custom data
# t2o = SSOF.oversamp_interp_helper(SSOF.bounds_generator(log_λ_obs), model.tel.log_λ)
# t2o = SSOF.undersamp_interp_helper(log_λ_obs, model.tel.log_λ)
# flux_tellurics = SSOF.spectra_interp(flux_tellurics, t2o)
plot(flux_tellurics[:, 1])
scatter(model_scores_telluric')

# # One could use barycorrpy to get realistic barycentric correction RVs
# # these would be on observations of HD 26965 from Kitt Peak
# using PyCall
# barycorrpy = PyNULL()
# # pyimport_conda("scipy", "scipy")
# # pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)
# copy!(barycorrpy , pyimport("barycorrpy") )
# barycentric_corrections = barycorrpy.get_BC_vel(JDUTC=times, hip_id=19849, obsname="KPNO", ephemeris="de430")
# barycentric_rvs = barycentric_corrections[1]

# we'll just use an epicyclic approximation for simplicity (correct within 411 m/s)
year = 365.25
phases = times ./ year * 2 * π
design_matrix = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
epicyclic_barycentric_rv_coefficients = [-20906.74122340397, -15770.355782489662, -390.29975114321905, -198.97407208182858, -67.99370656806558]
barycentric_rvs = design_matrix * epicyclic_barycentric_rv_coefficients

# Adding whatever additional RVs we want
injected_rvs = zeros(n_simulated_observations)
# K = 10  # planet velocity semi-amplitude (m/s)
# P = 60  # planet period (d)
# injected_rvs = K .* sin(times ./ P * 2 * π)


bandwidth_stellar = KernelDensity.default_bandwidth(vec(model.star.lm.s))
model_scores_stellar = rand.(Normal.(sample(vec(model.star.lm.s), n_simulated_observations; replace=true), bandwidth_stellar))'

log_λ_stellar = log_λ_obs .+ SSOF.rv_to_D(barycentric_rvs)'
_flux_stellar = SSOF._eval_lm(model.star.lm.M, model_scores_stellar, model.star.lm.μ; log_lm=model.star.lm.log)  # replace s for custom data
b2o = SSOF.StellarInterpolationHelper(model.star.log_λ, injected_rvs + barycentric_rvs, log_λ_obs)
flux_stellar = SSOF.spectra_interp(_flux_stellar, injected_rvs + barycentric_rvs, b2o)
flux_total = SSOF.total_model(flux_tellurics, flux_stellar)

## getting a realistic LSF
@load "examples/jld2/data.jld2" data
include("../NEID/lsf.jl")  # defines NEIDLSF.NEID_lsf()
lsf_simulated = NEIDLSF.neid_lsf(81, vec(mean(data.log_λ_obs; dims=2)), vec(mean(log_λ_obs; dims=2)))
flux_total_lsf = lsf_simulated * flux_total

# # getting a realistic blaze function for proper variance scaling
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
# @save "examples/jld2/blaze.jld2" blaze_function
@load "examples/jld2/blaze.jld2" blaze_function
var_total = flux_total_lsf ./ blaze_function.(log_λ_obs)

# SNR of center pixels
cp = Int(round(size(flux_total_lsf,1)/2))
median(flux_total_lsf[(cp-500):(cp+500),:] ./ sqrt.(var_total[(cp-500):(cp+500),:]))

flux_noisy = flux_total_lsf .+ (randn(size(var_total)) .* sqrt.(var_total))
data_simulated = SSOF.LSFData(flux_noisy, var_total, copy(var_total), log_λ_obs, log_λ_stellar, lsf_simulated) 
SSOF.mask_bad_edges!(data_simulated)

# getting initial model
star = "26965"
instrument = "SSOF"
order = 81
n_comp = 2  # how many components to use in the model

model_new = SSOF.calculate_initial_model(data_simulated;
	instrument=instrument, desired_order=order, star=star, times=times;
	max_n_tel=n_comp, max_n_star=n_comp, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
	log_λ_gp_tel=5.134684755671457e-6,
	tel_log_λ=model.tel.log_λ, star_log_λ=model.star.log_λ, oversamp=false)

# a plot
include("../SSOFUtilities/plots.jl")
df_act = Dict([("Input", vec(model_scores_stellar)), ("Input_σ", zeros(length(model_scores_stellar)))])
plot_model(model_new, vec(model_scores_telluric), times; df_act=df_act)

# setting up model workspace
mws = SSOF.ModelWorkspace(model_new, data_simulated)

# fitting regularization
SSOF.fit_regularization!(mws)

# performing final fit
results = SSOF.improve_model!(mws; iter=500, verbose=true, careful_first_step=true, speed_up=false)

# getting error bars
rvs, rvs_σ, tel_s_σ, star_s_σ = SSOF.estimate_σ_curvature(mws) # ~75% of bootstrap error bars
# rvs2, rvs2_σ, tel2_s_σ, star2_s_σ = SSOF.estimate_σ_bootstrap(mws)

## Plots

# how do our RV estimates look
plt = plot_model_rvs(times, rvs, rvs_σ, times, injected_rvs, zeros(length(times)); display_plt=true, title="SSOF", inst_str="Injected")
# plt = plot_model_rvs(times, rvs2, rvs2_σ, times, injected_rvs, zeros(length(times)); display_plt=true, title="SSOF", inst_str="Simulated")

# how well did the model fit the data
plt = status_plot(mws; display_plt=true)

# how close is each model to recreating the noiseless version used to simulate the data
plot(model.tel.λ, vec(std(flux_tellurics - model_new.tel.lm(); dims=2)); label="", xlabel = "Wavelength (Å)", ylabel = L"\textrm{std}(Y_\oplus - Y_\oplus^{\texttt{SSOF}})")
plot(model.star.λ, vec(std(_flux_stellar - model_new.star.lm(); dims=2)); label="", xlabel = "Wavelength (Å)", ylabel = L"\textrm{std}(Y_\star - Y_\star^{\texttt{SSOF}})")
# plt = _plot()
# plot_telluric_with_lsf!(plt[1], model_new, vec(std(flux_tellurics - model_new.tel.lm(); dims=2)); d=data_simulated, label="", color=plt_colors[1], alpha=0.7, legend=:outerright, xlabel = "Wavelength (Å)", ylabel = L"\textrm{std}(Y_\oplus - Y_\oplus^{\texttt{SSOF}})")
# plt = _plot()
# plot_stellar_with_lsf!(plt[1], model_new, vec(std(_flux_stellar - model_new.star.lm(); dims=2)); d=data_simulated, label="", color=plt_colors[1], alpha=0.7, legend=:outerright, xlabel = "Wavelength (Å)", ylabel = L"\textrm{std}(Y_\star - Y_\star^{\texttt{SSOF}})")


# how close is each template and feature vector to the source models
function compare_model_component(source, model, stellar::Bool; kwargs...)
	plt = _plot(; layout = grid(2, 1, heights=[0.75, 0.25]), kwargs...)
	stellar ? plot_f = plot_stellar_with_lsf! : plot_f = plot_telluric_with_lsf!
	plot_f(plt[1], model_new, source; d=data_simulated, label="Source", color=plt_colors[1], alpha=0.7, legend=:outerright, ylabel = "Continuum Normalized Flux + Const")
	plot_f(plt[1], model_new, model .- (maximum(model) - minimum(source)); d=data_simulated, label="Model", color=plt_colors[2], alpha=0.7)
	plot_f(plt[2], model_new, source .- model; d=data_simulated, label="", color=base_color, alpha=0.7, legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Residual", title="")
	return plt
end

compare_model_component(model.tel.lm.μ,  model_new.tel.lm.μ, false; title=L"\mu_\oplus")
compare_model_component(model.tel.lm.M[:, 1] ./ norm(model.tel.lm.M[:, 1]),  model_new.tel.lm.M[:, 1] ./ norm(model_new.tel.lm.M[:, 1]), false; title=L"W_\oplus")

compare_model_component(model.star.lm.μ,  model_new.star.lm.μ, true; title=L"\mu_\star")
compare_model_component(model.star.lm.M[:, 1]./ norm(model.star.lm.M[:, 1]),  model_new.star.lm.M[:, 1]./ norm(model_new.star.lm.M[:, 1]), true; title=L"W_\star")

# plot_telluric_bases(model_new, data_simulated)
# plot_telluric_bases(model, data_simulated)

# plot_stellar_bases(model_new, data_simulated)
# plot_stellar_bases(model, data_simulated)

# how close are the measuresd scores to the source scores
normalized(x) = (x .- mean(x)) / std(x)
function compare_model_score(source, model; kwargs...)
	plt = _plot(; layout = grid(2, 1, heights=[0.75, 0.25]), kwargs...)
	s = normalized(source)
	m = normalized(model)
	scatter!(plt[1], times, s; label="Source", alpha=0.7, legend=:outerright, ylabel = "Normalized Scores")
	scatter!(plt[1], times, m .- (maximum(m) - minimum(s)); label="Model", color=plt_colors[2], alpha=0.7)
	scatter!(plt[2], times, s .- m; label="", color=base_color, xlabel = "Wavelength (Å)", ylabel = "Residual", title="", legend=:outerright)
	return plt
end

compare_model_score(model_scores_telluric', model_new.tel.lm.s'; title=L"S_\oplus")
compare_model_score(model_scores_stellar', model_new.star.lm.s'; title=L"S_\star")
# scatter(normalized(model_scores_stellar)', normalized(model_scores_stellar)' .- normalized(model_new.star.lm.s)')