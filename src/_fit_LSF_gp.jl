# Finding reasonable LSF_gp param values from fitting to a lsf broadened line
# Based roughly on the EXPRES LSF at wn = 17000 1/cm (588 nm)
using AbstractGPs
using KernelFunctions
using TemporalGPs
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation

using Plots

wavenumber_to_Å(wn) = 1e8 ./ wn
_fwhm_2_σ_factor = 1 / (2 * sqrt(2 * log(2)))
fwhm_2_σ(fwhm::Real) = _fwhm_2_σ_factor * fwhm

# Ingesting data
central_wn = 17000
n = 1000
λ_lo, λ_hi = log(wavenumber_to_Å(central_wn+1/2)), log(wavenumber_to_Å(central_wn-1/2))
λs = RegularSpacing(λ_lo, (λ_hi-λ_lo)/n, n)
wns = wavenumber_to_Å.(exp.(λs))
expres_LSF_FWHM_wn = 0.14  # 1/cm
expres_LSF_σ_Å = fwhm_2_σ(0.14)
quiet = 1 .- exp.(-((wns .- wns[Int(round(n/2))]) ./ expres_LSF_σ_Å) .^ 2)

plot(λs, quiet)


# Declare model parameters using `ParameterHandling.jl` types.
flat_initial_params, unflatten = value_flatten((
	var_kernel = positive(1.),
	λ = positive(1e5),
))

params = unflatten(flat_initial_params)

function build_gp(params)
    f_naive = GP(params.var_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
    return to_sde(f_naive, SArrayStorage(Float64))
end

# Data
x = λs
y = quiet .- 1

# Changing this changes the results significantly
# ↑var_noise → ↑λ ↓var_kernel
# var_noise = 1e-4 seems to lead to most sensible results i.e. draws from the
# prior of the optimal result look similar to the input spectra
var_noise = 1e-4
function objective(params)
    f = build_gp(params)
    return -logpdf(f(x, var_noise), y)
end

# Check that the objective function works:
objective(params)

f = objective ∘ unflatten
function g!(G, θ)
	G .= only(Zygote.gradient(f, θ))
end
# f(flat_initial_params)
# G = zeros(length(flat_initial_params))
# g!(G, flat_initial_params)

training_results = optimize(f, g!, flat_initial_params,
	BFGS(alphaguess = Optim.LineSearches.InitialStatic(scaled=true),linesearch = Optim.LineSearches.BackTracking()),
	Optim.Options(show_trace=true))

final_params = unflatten(training_results.minimizer)
println(final_params)

f = build_gp(final_params)
fx = f(x, var_noise)
f_post = posterior(fx, y)
# Compute the posterior marginals.
y_post = marginals(f_post(x))


pli = 1:length(x)
ym = mean.(y_post[pli])
ys = std.(y_post[pli])

function plt(var_kernel, λ; plot_sample=true)
	pt = plot(x[pli], y[pli], label="data")
	plot!(pt, x[pli], ym, alpha=0.8, ribbon=(-ys,ys), label="fit posterior")

	params = (var_kernel = var_kernel,
		λ = λ,)
	f = build_gp(params)
	fx = f(x, var_noise)
	if plot_sample; plot!(pt, x[pli], rand(fx)[pli], label="sample from input") end
	f_post2 = posterior(fx, y)
	y_post2 = marginals(f_post2(x))
	ym2 = mean.(y_post2[pli])
	ys2 = std.(y_post2[pli])
	plot!(pt, x[pli], ym2, alpha=0.8, ribbon=(-ys2,ys2), label="input posterior")
	println(-logpdf(fx, y))
	pt
end
# plt(1, 7570)
plt(final_params.var_kernel, final_params.λ)


# n=30
# wavs = LinRange(log(1e3), log(1e6), n)
# vars = LinRange(log(1e-2), log(1e3), n)
# holder = zeros(n, n)
# function ℓ(vars, wavs; y=y)
# 	params = (var_kernel = exp(vars),
# 		λ = exp(wavs),)
# 	f = build_gp(params)
# 	fx = f(x, 1e-6)
# 	return -logpdf(fx, y)
# end
# for i in 1:n
# 	for j in 1:n
# 		holder[i,j] = ℓ(vars[j], wavs[i])
# 	end
# end
#
# ch = copy(holder)
# ch[ch .> 0] .= 0
# # heatmap(ch; xlabel="vars", ylabel="wavs")
# pt = heatmap(exp.(vars), exp.(wavs), ch; xlabel="vars", ylabel="wavs", xscale=:log10, yscale=:log10)
# png(pt, "tel gp params heatmap")
