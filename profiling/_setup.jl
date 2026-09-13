# Shared simulated-data + initial-model setup, factored out of
# examples/example.jl so profile_example.jl and bench_gradient.jl don't
# duplicate the simulation code. The calling script must already have `REPO`
# defined (repo root) and `SSOF`, `JLD2`, `Statistics`, `KernelDensity`,
# `Distributions`, `StatsBase` loaded.

include(joinpath(REPO, "examples", "_lsf.jl"))  # defines NEIDLSF.neid_lsf

function build_simulated_data(; n_simulated_observations::Int=50, desired_max_snr::Real=500)
	data_dir = joinpath(REPO, "examples", "data")
	@load joinpath(data_dir, "results.jld2") model
	@load joinpath(data_dir, "data.jld2") data

	injected_rvs = zeros(n_simulated_observations)
	times = 2459580 .+ collect(LinRange(365., 0, n_simulated_observations))

	year = 365.25
	phases = times ./ year * 2 * π
	design_matrix = hcat(cos.(phases), sin.(phases), cos.(2 .* phases), sin.(2 .* phases), ones(length(times)))
	epicyclic_barycentric_rv_coefficients = [-20906.74122340397, -15770.355782489662, -390.29975114321905, -198.97407208182858, -67.99370656806558]
	barycentric_rvs = design_matrix * epicyclic_barycentric_rv_coefficients

	log_λ_obs = model.tel.log_λ * (ones(n_simulated_observations)')
	log_λ_stellar = log_λ_obs .+ (SSOF.rv_to_D(barycentric_rvs)')

	bandwidth_stellar = KernelDensity.default_bandwidth(vec(model.star.lm.s))
	model_scores_stellar = rand.(Normal.(sample(vec(model.star.lm.s), n_simulated_observations; replace=true), bandwidth_stellar))'
	model_scores_stellar .-= mean(model_scores_stellar)
	_flux_stellar = SSOF._eval_lm(model.star.lm.M, model_scores_stellar, model.star.lm.μ; log_lm=model.star.lm.log)

	b2o = SSOF.StellarInterpolationHelper(model.star.log_λ, injected_rvs + barycentric_rvs, log_λ_obs)
	flux_stellar = SSOF.spectra_interp(_flux_stellar, injected_rvs + barycentric_rvs, b2o)

	bandwidth_telluric = KernelDensity.default_bandwidth(vec(model.tel.lm.s))
	model_scores_telluric = rand.(Normal.(sample(vec(model.tel.lm.s), n_simulated_observations; replace=true), bandwidth_telluric))'
	model_scores_telluric .-= mean(model_scores_telluric)
	flux_tellurics = SSOF._eval_lm(model.tel.lm.M, model_scores_telluric, model.tel.lm.μ; log_lm=model.tel.lm.log)

	flux_total = SSOF.total_model(flux_tellurics, flux_stellar)

	lsf_simulated = NEIDLSF.neid_lsf(81, vec(mean(data.log_λ_obs; dims=2)), vec(mean(log_λ_obs; dims=2)))
	flux_total_lsf = lsf_simulated * flux_total

	@load joinpath(data_dir, "blaze.jld2") blaze_function
	var_total = flux_total_lsf ./ blaze_function.(log_λ_obs)

	cp = Int(round(size(flux_total_lsf, 1) / 2))
	snr_cp = median(flux_total_lsf[(cp-500):(cp+500), :] ./ sqrt.(var_total[(cp-500):(cp+500), :]))
	var_total .*= (snr_cp / desired_max_snr)^2

	flux_noisy = flux_total_lsf .+ (randn(size(var_total)) .* sqrt.(var_total))

	data_simulated = SSOF.LSFData(flux_noisy, var_total, copy(var_total), log_λ_obs, log_λ_stellar, lsf_simulated)
	SSOF.mask_bad_edges!(data_simulated)

	return model, data_simulated, times
end

function build_initial_model(model, data_simulated, times; n_comp::Int=2)
	return SSOF.calculate_initial_model(data_simulated;
		instrument="SSOF", desired_order=81, star="26965", times=times,
		max_n_tel=n_comp, max_n_star=n_comp, log_λ_gp_star=1/SSOF.SOAP_gp_params.λ,
		log_λ_gp_tel=5.134684755671457e-6,
		tel_log_λ=model.tel.log_λ, star_log_λ=model.star.log_λ, oversamp=false)
end
