using Pkg
Pkg.activate(".")

# Finding reasonable LSF_gp param values from fitting to a lsf broadened line
# Based roughly on the EXPRES LSF at wn = 17000 1/cm (588 nm)
using AbstractGPs
using KernelFunctions
using TemporalGPs
using Optim # Standard optimisation algorithms.
using ParameterHandling # Helper functionality for dealing with model parameters.
using Zygote # Algorithmic Differentiation
import StellarSpectraObservationFitting as SSOF

using Plots

wavenumber_to_Å(wn) = 1e8 ./ wn
_fwhm_2_σ_factor = 1 / (2 * sqrt(2 * log(2)))
fwhm_2_σ(fwhm::Real) = _fwhm_2_σ_factor * fwhm
n = 1000
function matern52_kernel_base(λ::Number, δ::Number)
    x = sqrt(5) * abs(δ) / λ
    return (1 + x * (1 + x / 3)) * exp(-x)
end
plot_mat_λ!(λ, x) = plot!(x, matern52_kernel_base.((1 / λ), x); label="λ=$λ")
function fits_lsfs(sim_step, sep_test, quiet::Vector, folder::String, lsf_λs::Vector, i::Int, order::Int)
	cov_xs = sim_step .* sep_test
	# covs = [cor(quiet[1:end-sep], quiet[1+sep:end]) for sep in sep_test]
	# covs = [cov(quiet[1:end-sep], quiet[1+sep:end]) for sep in sep_test]
	# covs ./= maximum(covs)
	covs = [quiet[1:end-sep]'*quiet[1+sep:end] for sep in sep_test]  # better covariance function that knows they should be zero mean over long baseline
	covs ./= maximum(covs)
	plot(cov_xs, covs; label="lsf line", title="How correlated are nearby wavelengths", ylabel="cov", xlabel="log(λ (Å))")
	# plot!(cov_xs, covs; label="better cov")
	plot_mat_λ!(1e5, cov_xs)
	plot_mat_λ!(2e5, cov_xs)
	plot_mat_λ!(3e5, cov_xs)
	# weights = max.(covs, 0)
	# fo(λ) = sum(weights .* ((covs - matern52_kernel_base.((1 / λ[1]), cov_xs)) .^ 2))
	fo(λ) = sum((covs - matern52_kernel_base.((1 / λ[1]), cov_xs)) .^ 2)
	result = optimize(fo, [2e5], LBFGS(); autodiff = :forward)
	lsf_λs[i] = result.minimizer[1]
	plot_mat_λ!(round(result.minimizer[1]), cov_xs)
	png("figs/lsf/$folder/$(order)_lsf_cor")
end

## EXPRES
using CSV, DataFrames

_eo = CSV.read("EXPRES/expres_psf.txt", DataFrame)
# eo = CSV.read("C:/Users/chris/Downloads/expres_psf.txt", DataFrame)
filter!(:line => ==("LFC"), _eo)
sort!(_eo, ["wavenumber [1/cm]"])

# has orders 37:76, (neid's 50-89), 41-76 is the intersection
lsf_orders = 37:75
lsf_λs = Array{Float64}(undef, length(lsf_orders))
for i in eachindex(lsf_orders)
	order = lsf_orders[i]
	eo = copy(_eo)
	filter!(:order => ==(order), eo)
	middle = Int(round(nrow(eo)/2))
	central_wn = mean(eo."wavenumber [1/cm]"[middle-10:middle+10])
	expres_LSF_FWHM_wn = mean(eo."fwhm [1/cm]"[middle-10:middle+10])
	λ_lo, λ_hi = log(wavenumber_to_Å(central_wn+4expres_LSF_FWHM_wn)), log(wavenumber_to_Å(central_wn-4expres_LSF_FWHM_wn))
	λs = RegularSpacing(λ_lo, (λ_hi-λ_lo)/n, n)
	wns = wavenumber_to_Å.(exp.(λs))
	expres_LSF_σ_wn = fwhm_2_σ(expres_LSF_FWHM_wn)
	quiet = exp.(-(((wns .- wns[Int(round(n/2))]) ./ expres_LSF_σ_wn) .^ 2)/2)
	# quiet .-= mean(quiet)
	# plot!(λs .+ mean_λ, quiet)

	# how many λs.Δt does it take to contain ~2σ?
	n_seps = ((λ_hi + λ_lo) / 2 - log(wavenumber_to_Å(wavenumber_to_Å(exp((λ_hi + λ_lo) / 2)) + 2expres_LSF_σ_wn))) / λs.Δt
	sep_test = 0:Int(ceil(n_seps))
	sim_step = λs.Δt
	fits_lsfs(sim_step, sep_test, quiet, "expres", lsf_λs, i, order)
end
maximum(lsf_λs)

lsf_λs_smooth = Array{Float64}(undef, length(lsf_orders))
f1 = SSOF.ordinary_lst_sq_f(lsf_λs, 2)
lsf_λs_smooth .= f1.(1.:39)
println(lsf_λs_smooth)

plot(lsf_λs)
plot!(lsf_λs_smooth)

## NEID
using JLD2

# Ingesting data (using NEID LSF)
include("../NEID/lsf.jl")  # defines NEIDLSF.NEID_lsf()
nlsf = NEIDLSF
npix = 30
@load "order_pixel_spacing.jld2" spacing lsf_orders
lsf_λs = Array{Float64}(undef, length(lsf_orders))
middle = Int(round(size(nlsf.σs,1)/2))
pix = RegularSpacing(-1. * npix, 2npix/n, n)
for i in eachindex(lsf_orders)
# i = 1  # 1:59
	order = lsf_orders[i] # has orders 54:112, (expres's 41-99), 54-89 is the intersection (index 1 and 36)
	λs = RegularSpacing(-npix * spacing[i], 2*npix*spacing[i]/n, n)
	quiet = nlsf.conv_gauss_tophat.(pix, nlsf.σs[middle, order], nlsf.bhws[middle, order])
	quiet ./= maximum(quiet)

	# how many λs.Δt does it take to contain ~2σ?
	tot = 0
	# target = sum(quiet)*0.841
	# target = sum(quiet)*0.977
	# target = sum(quiet)* 0.9938
	target = sum(quiet)* (1-3.167e-5)
	j = 0
	while tot < target
		j += 1
		tot += quiet[j]
	end
	sep_test = 0:(j-500+1)
	sim_step = λs.Δt
	fits_lsfs(sim_step, sep_test, quiet, "neid", lsf_λs, i, order)
end
maximum(lsf_λs)
lsf_λs_smooth = Array{Float64}(undef, length(lsf_orders))
f1 = SSOF.ordinary_lst_sq_f(lsf_λs[1:35], 2)
lsf_λs_smooth[1:35] .= f1.(1.:35)
f2 = SSOF.ordinary_lst_sq_f(lsf_λs[35:59], 2;x=35.:59)
lsf_λs_smooth[36:59] .= f2.(36.:59)
println(lsf_λs_smooth)

plot(lsf_λs)
plot!(lsf_λs_smooth)

## Visual inspection

@load "order_pixel_spacing.jld2" spacing lsf_orders
middle = Int(round(size(nlsf.σs,1)/2))
pix = RegularSpacing(-1. * npix, 2npix/n, n)
i = 1
order = lsf_orders[30] # has orders 54:112, (expres's 41-99), 54-89 is the intersection (index 1 and 36)
λs = RegularSpacing(-npix * spacing[i], 2*npix*spacing[i]/n, n)
quiet = nlsf.conv_gauss_tophat.(pix, nlsf.σs[middle, order], nlsf.bhws[middle, order])
quiet ./= maximum(quiet)

function plt(var_kernel, λ; plot_sample=true)
	pt = plot(λs, quiet, label="data")
	f = build_gp((var_kernel = var_kernel, λ = λ))
	fx = f(λs, 1e-6)
	if plot_sample
		for i in 1:3
			plot!(pt, λs, rand(fx), label="sample")
		end
	end
	f_post2 = posterior(fx, quiet)
	y_post2 = marginals(f_post2(λs))
	ym2 = mean.(y_post2)
	ys2 = std.(y_post2)
	plot!(pt, λs, ym2, alpha=0.8, ribbon=(-ys2,ys2), label="posterior")
	return pt
end
plt(0.2, 180000.; plot_sample=true)
