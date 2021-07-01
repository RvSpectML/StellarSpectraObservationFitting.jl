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

# orders[[i for i in 1:length(orders) if abs(rvs[i, 1]) > 100]]
inds = orders2inds(orders[1:end-6])

@load "$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

# Compare RV differences to actual RVs from activity
plt = plot_model_rvs_new(times_nu, rvs_red, rvs_σ_red, eo_time, eo_rv, eo_rv_σ; markerstrokewidth=1)
png(plt, star * "_model_rvs.png")
# end
# using Distributions
#
# function helper(x::Real, i::Int)
#     ans = 0
#     for j in 1:length(selected_orders)
#         ans += pdf(Distributions.Normal(rvs[j, i], rvs_σ[j, i]), x)
#     end
#     return ans / length(selected_orders)
# end
# helper(xs::AbstractVector, i::Int) = [helper(x, i) for x in xs]
# x = LinRange(-10,10,1000)
# plot(x, helper(x, 1); label = "model makeup")
# plot!(x, pdf.(Distributions.Normal(eo_rv[1], eo_rv_σ[1]), x); label = "EXPRES")
# plot!(x, pdf.(Distributions.Normal(rvs_red[1], rvs_σ_red[1]), x); label = "model")


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
