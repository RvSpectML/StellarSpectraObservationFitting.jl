## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using CSV, DataFrames

## Setting up necessary variables

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
stars = ["10700", "26965"]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]
orders_list = [42:77, 40:77]

# for star_ind in 1:2
star_ind = SSOF.parse_args(1, Int, 2)
star = stars[star_ind]
orders = orders_list[star_ind]

## Looking at model components

@load "$(star)_md.jld2" n_comps n_comps_bic robust

n_robust = [!i for i in robust]
x = orders_list[star_ind]
annot=text.(x, :top, :white, 9)
α = 1
# robust_str = ["" for i in x]
# for i in 1:length(robust_str)
#     robust_str[i] *= "$(x[i])"
#     if !robust[i]; robust_str[i] *= "!" end
# end
# annot=text.(robust_str, :top, :white, 9)
plt = _my_plot(; ylabel="# of basis vectors", xlabel="Order", title="Best Models for $star (Based on AIC)", xticks=false)
my_scatter!(plt, x, n_comps[:, 1]; alpha=α, label="# of telluric components", legend=:topleft, series_annotations=annot)
my_scatter!(plt, x, n_comps[:, 2]; alpha=α, label="# of stellar components", series_annotations=annot)
plot!(plt, x, n_comps[:, 1]; label = "", alpha=α, color=plt_colors[1], ls=:dot)
plot!(plt, x, n_comps[:, 2]; label = "", alpha=α, color=plt_colors[2], ls=:dot)
my_scatter!(plt, x[n_robust], n_comps_bic[n_robust, 1]; alpha=α/2, color=plt_colors[11], label="# of telluric components (BIC)")
my_scatter!(plt, x[n_robust], n_comps_bic[n_robust, 2]; alpha=α/2, color=plt_colors[12], label="# of stellar components (BIC)")
plot!(plt, x, n_comps_bic[:, 1]; label = "", alpha=α/2, color=plt_colors[11], ls=:dot)
plot!(plt, x, n_comps_bic[:, 2]; label = "", alpha=α/2, color=plt_colors[12], ls=:dot)

png(plt, "md_$star.png")

## Comparing to CCF RVs

@load "EXPRES\\alex_stuff\\HD$(star)q0f0n1w1e=false_order_results.jld2" rvs_ccf_orders good_orders order_weights
good_orders_mask = [i in orders for i in 12:83] .& good_orders
good_orders_2 = (12:83)[good_orders_mask]
bad_orders_2 = (12:83)[[i in orders for i in 12:83] .& .!(good_orders)]
plt = my_scatter(12:83,order_weights; label="", title="$star Order Weights", xlabel="Orders", c=[i ? :green : :red for i in good_orders])
png(plt, "order_weights")

using Plots.PlotMeasures

myplt(x, y, z) = heatmap(x, y, z; size=(600,400), right_margin=20px, ylabel="orders", xlabel="obs", title="HD"*star)
# plt = myplt(1:size(rvs_ccf_orders,2), 12:83, rvs_ccf_orders)
plt = myplt(1:size(rvs_ccf_orders,2), (12:83)[good_orders], rvs_ccf_orders[good_orders, :] .- median(rvs_ccf_orders[good_orders, :]; dims=2))
png(plt, "test1")
ccf_rvs = rvs_ccf_orders[good_orders_mask, :]
plt = myplt(1:size(ccf_rvs,2), (12:83)[good_orders_mask], ccf_rvs)
png(plt, "test2")
ccf_rvs .-= median(ccf_rvs; dims=2)
heatmap(ccf_rvs)

## RV reduction

@load "$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
# # plotting order means which don't matter because the are constant shifts for the reduced rv
# my_scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= mean(rvs; dims=2)

plt = my_scatter(orders, std(rvs; dims=2); legend=:topleft, label="", title="$star RV std", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
png(plt, "order_rv_std")
plt = my_scatter(orders, median(rvs_σ; dims=2); legend=:topleft, label="", title="$star Median σ", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
png(plt, "order_rv_σ")
plt = my_scatter(orders, std(rvs; dims=2) ./ median(rvs_σ; dims=2); legend=:topleft, label="", title="$star (RV std) / (Median σ)", xlabel="Order", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
png(plt, "order_rv_ratio")
scatter(orders, sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2); label="χ²", legend=:topleft)

inds = orders2inds(orders[1:end-2])
# inds = orders2inds(good_orders)

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

# Compare RV differences to actual RVs from activity
plt = plot_model_rvs_new(times_nu, -rvs_red, rvs_σ_red, eo_time, eo_rv, eo_rv_σ; markerstrokewidth=1)
png(plt, star * "_model_rvs.png")
# end

Pkg.add("Distributions")
using Distributions

selected_orders = good_orders

function helper(x::Real, i::Int)
    ans = 0
    for j in 1:length(selected_orders)
        ans += pdf(Distributions.Normal(rvs[j, i], rvs_σ[j, i]), x)
    end
    return ans / length(selected_orders)
end
helper(xs::AbstractVector, i::Int) = [helper(x, i) for x in xs]
x = LinRange(-20,20,1000)
u = 14
plot(x, helper(x, u); label = "model makeup")
plot!(x, pdf.(Distributions.Normal(eo_rv[u], eo_rv_σ[u]), x); label = "EXPRES")
plot!(x, pdf.(Distributions.Normal(rvs_red[u], rvs_σ_red[u]), x); label = "model")


## Periodograms

import GLOM_RV_Example; GLOM_RV = GLOM_RV_Example
import GPLinearODEMaker; GLOM = GPLinearODEMaker
using Unitful
Σ_obs = Diagonal(rvs_σ2_red)
times = times_nu .* u"d"

function fit_kep_hold_P(P::Unitful.Time; fast::Bool=false, kwargs...)
    #initialize with fast epicyclic fit
    ks = GLOM_RV.fit_kepler(rvs_red, times, Σ_obs, GLOM_RV.kep_signal_epicyclic(P=P))
    if !fast
        ks = GLOM_RV.fit_kepler(rvs_red, times, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, ks.M0, minimum([ks.e, 0.3]), 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
        return ks
    end
    if ks == nothing
        ks = GLOM_RV.fit_kepler(rvs_red, times, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, 2 * π * rand(), 0.1, 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
        return ks
    end
    return ks
end

function kep_unnormalized_posterior_distributed(P::Unitful.Time; kwargs...)
    ks = fit_kep_hold_P(P; kwargs...)
    nlogprior_kernel = 0
    if ks == nothing
        return [-Inf, -Inf]
    else
        val = GLOM.nlogL(Σ_obs, GLOM_RV.remove_kepler(rvs_red, times, ks))
        return [-val, GLOM_RV.logprior_kepler(ks; use_hk=false) - nlogprior_kernel - val]
    end
end

freq_grid = GLOM_RV.autofrequency(times; samples_per_peak=6)
freq_grid = GLOM_RV.autofrequency(times; samples_per_peak=11)
period_grid = 1 ./ reverse(freq_grid)
amount_of_periods = length(period_grid)

likelihoods = zeros(amount_of_periods)
unnorm_posteriors = zeros(amount_of_periods)

# takes around minutes for 101501 data and 3000 periods
@progress for i in 1:100
    likelihoods[i], unnorm_posteriors[i] = kep_unnormalized_posterior_distributed(period_grid[i])
end

best_periods = period_grid[GLOM_RV.find_modes(unnorm_posteriors; amount=10)]
best_period = best_periods[1]

println("found period:    $(ustrip(best_period)) days")

plot(ustrip.(period_grid), likelihoods; xaxis=:log, leg=false)
png(star * "period_lik")
plot(ustrip.(period_grid), unnorm_posteriors; xaxis=:log, leg=false)
png(star * "period_evi")
