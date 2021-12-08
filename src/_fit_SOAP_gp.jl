# Finding reasonable SOAP_gp param values from fitting to a quiet solar spectrum
#  downloaded from https://zenodo.org/record/3753254
using HDF5
using AbstractGPs
using KernelFunctions
using TemporalGPs
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation
using ParameterHandling

# Ingesting data
hdf5_loc = "C:/Users/chris/Downloads/res-1000-1years_full_id1.h5"
fid = h5open(hdf5_loc, "r")
quiet = fid["quiet"][:]
inds = quiet .!= 0
λs = fid["lambdas"][:]#u"nm"/10

λs = log.(λs[inds])
quiet = quiet[inds]
quiet ./= maximum(quiet)

std(y)
# using Plots
# plot(λs, quiet)

# Setting up kernel
use_matern = false
if use_matern
	flat_initial_params, unflatten = value_flatten((
		σ²_kernel = positive(0.1),
		λ = positive(4e4),
		))
	function build_gp(params)
	    f_naive = GP(params.σ²_kernel * Matern52Kernel() ∘ ScaleTransform(params.λ))
	    return to_sde(f_naive, SArrayStorage(Float64))
	end
else
	flat_initial_params, unflatten = value_flatten((
		σ²_kernel = positive(0.1),
		λ = positive(1e4),
		))
	function build_gp(params)
	    f_naive = GP(params.σ²_kernel * PiecewisePolynomialKernel(;degree=2, dim=1) ∘ ScaleTransform(params.λ))
		# return f_naive
	    return to_sde(f_naive)
	end
end
params = unflatten(flat_initial_params)

# Data
x = λs
y = quiet .- 1
# Changing this changes the results significantly
# ↑σ²_noise → ↑λ ↓σ²_kernel
# σ²_noise = 1e-6 seems to lead to most sensible results i.e. draws from the
# prior of the optimal result look similar to the input spectra
σ²_noise = 1e-6
function objective(params)
    f = build_gp(params)
    return -logpdf(f(x, σ²_noise), y)
end

# Check that the objective function works:
objective(params)

f = objective ∘ unflatten
function g!(G, θ)
	G .= only(Zygote.gradient(f, θ))
end

training_results = optimize(f, g!, flat_initial_params,
	BFGS(alphaguess = Optim.LineSearches.InitialStatic(scaled=true),linesearch = Optim.LineSearches.BackTracking()),
	Optim.Options(store_trace=true, show_trace=false))
final_params = unflatten(training_results.minimizer)
println(final_params)

# f = build_gp(final_params)
# fx = f(x, final_params.σ²_noise)
# f_post = posterior(fx, y)
# # Compute the posterior marginals.
# y_post = marginals(f_post(x))
#
#
# using Plots
# pli = 101000:102000
# ym = mean.(y_post[pli])
# ys = std.(y_post[pli])
#
# function plt(σ²_kernel, λ; plot_sample=true)
# 	plot(x[pli], y[pli])
# 	plot!(x[pli], ym, alpha=0.8, ribbon=(-ys,ys))
#
# 	params = (σ²_kernel = σ²_kernel,
# 		λ = λ,)
# 	f = build_gp(params)
# 	fx = f(x, σ²_noise)
# 	if plot_sample; plot!(x[pli], rand(fx)[pli]) end
# 	f_post2 = posterior(fx, y)
# 	y_post2 = marginals(f_post2(x))
# 	ym2 = mean.(y_post2[pli])
# 	ys2 = std.(y_post2[pli])
# 	plot!(x[pli], ym2, alpha=0.8, ribbon=(-ys2,ys2))
# 	println(-logpdf(fx, y))
# end
# plt(0.1, 4e4)
# plt(0.001, 4e4; plot_sample=false)
# plt(1e3, 1e4; plot_sample=false)
# plt(final_params.σ²_kernel, final_params.λ)
#
# n=30
# wavs = LinRange(log(1e3), log(2e5), n)
# vars = LinRange(log(1e-3), log(1e3), n)
# holder = zeros(n, n)
# for i in 1:n
# 	for j in 1:n
# 		holder[i,j] = ℓ(vars[j], wavs[i])
# 	end
# end
# function ℓ(vars, wavs; y=y)
# 	params = (σ²_kernel = exp(vars),
# 		λ = exp(wavs),)
# 	f = build_gp(params)
# 	fx = f(x, 1e-6)
# 	return -logpdf(fx, y)
# end
# ch = copy(holder)
# ch[ch .> 0] .= 0
# heatmap(ch; xlabel="vars", ylabel="wavs")
# heatmap(exp.(vars), exp.(wavs), ch; xlabel="vars", ylabel="wavs", xscale=:log10, yscale=:log10)
