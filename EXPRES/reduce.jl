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
stars = ["10700", "26965", "34411"]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]
orders_list = [1:85, 1:85, 1:85]
prep_str = "noreg_"
prep_str = ""

# for star_ind in 1:2
star_ind = SSOF.parse_args(1, Int, 3)
star = stars[star_ind]
orders = orders_list[star_ind]

## Looking at model components

@load "$(prep_str)$(star)_md.jld2" n_comps n_comps_bic robust

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

png(plt, "$(prep_str)md_$star.png")

# ## Comparing to CCF RVs
#
# @load "EXPRES\\alex_stuff\\HD$(star)q0f0n1w1e=false_order_results.jld2" rvs_ccf_orders good_orders order_weights
# good_orders_mask = [i in orders for i in 12:83] .& good_orders
# good_orders_2 = (12:83)[good_orders_mask]
# bad_orders_2 = (12:83)[[i in orders for i in 12:83] .& .!(good_orders)]
# plt = my_scatter(12:83,order_weights; label="", title="$star Order Weights", xlabel="Orders", c=[i ? :green : :red for i in good_orders])
# png(plt, "order_weights")
#
# using Plots.PlotMeasures
#
# myplt(x, y, z) = heatmap(x, y, z; size=(600,400), right_margin=20px, ylabel="orders", xlabel="obs", title="HD"*star)
# # plt = myplt(1:size(rvs_ccf_orders,2), 12:83, rvs_ccf_orders)
# plt = myplt(1:size(rvs_ccf_orders,2), (12:83)[good_orders], rvs_ccf_orders[good_orders, :] .- median(rvs_ccf_orders[good_orders, :]; dims=2))
# png(plt, "test1")
# ccf_rvs = rvs_ccf_orders[good_orders_mask, :]
# plt = myplt(1:size(ccf_rvs,2), (12:83)[good_orders_mask], ccf_rvs)
# png(plt, "test2")
# ccf_rvs .-= median(ccf_rvs; dims=2)
# heatmap(ccf_rvs)


## RV reduction

@load "$(prep_str)$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
# # plotting order means which don't matter because the are constant shifts for the reduced rv
# my_scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= median(rvs; dims=2)

# plt = my_scatter(orders, std(rvs; dims=2); legend=:topleft, label="", title="$star RV std", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
# png(plt, prep_str * star * "_order_rv_std")
# plt = my_scatter(orders, median(rvs_σ; dims=2); legend=:topleft, label="", title="$star Median σ", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
# png(plt, prep_str * star * "_order_rv_σ")
# plt = my_scatter(orders, std(rvs; dims=2) ./ median(rvs_σ; dims=2); legend=:topleft, label="", title="$star (RV std) / (Median σ)", xlabel="Order", size=(_plt_size[1]*0.5,_plt_size[2]*0.75))
# png(plt, prep_str * star * "_order_rv_ratio")
χ² = vec(sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2))
annot=text.(orders[sortperm(χ²)], :top, :white, 9)
plt = my_scatter(1:length(χ²), sort(χ²); label="χ²", series_annotations=annot, legend=:topleft, title=prep_str * star * "_χ²") #, yaxis=:log)
png(plt, prep_str * star * "_χ²")

# inds = orders2inds(orders[1:end-2])
inds = sort(sortperm(χ²)[1:end-5])
# inds = orders2inds(good_orders)

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_red .-= median(rvs_red)
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv .-= median(eo_rv)
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

ccf_output = CSV.read(SSOF_path * "\\EXPRES\\alex_stuff\\HD$(star)q0f0n1w1e=false_RVs.csv", DataFrame; header=false)
ccf_rvs = Array(ccf_output[1, :])
ccf_rvs .-= median(ccf_rvs)

# Compare RV differences to actual RVs from activity
plt = plot_model_rvs_new(times_nu, rvs_red, rvs_σ_red, eo_time, eo_rv, eo_rv_σ, ccf_rvs; markerstrokewidth=1, title="HD"*star)
png(plt, prep_str * star * "_model_rvs.png")
# end
