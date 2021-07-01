## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

stars = ["10700", "26965"]
orders_list = [42:77, 40:77]
star_ind = SSOF.parse_args(1, Int, 2)
star = stars[star_ind]
orders = orders_list[star_ind]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]

inds = orders2inds(orders[1:end-6])

@load "$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

rvs_red = (sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'
rvs_σ_red = 1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")

using CSV, DataFrames
expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

# Compare RV differences to actual RVs from activity
plt = plot_model_rvs_new(times_nu, rvs_red, rvs_σ_red, eo_time, eo_rv, eo_rv_σ; markerstrokewidth=1)
png(plt, star * "_model_rvs.png")

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
