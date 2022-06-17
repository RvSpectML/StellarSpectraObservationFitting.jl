## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using CSV, DataFrames
using Statistics
using Plots

## Setting up necessary variables

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
stars = ["10700", "26965", "34411"]
orders_list = repeat([1:85], length(stars))
include("data_locs.jl")  # defines expres_data_path and expres_save_path
star_ind = SSOF.parse_args(1, Int, 2)
dpca = SSOF.parse_args(2, Bool, false)
if dpca
    prep_str = ""
else
    prep_str = "wobble_"
end
only_excalibur = SSOF.parse_args(3, Bool, true)
# excals = Bool.(zeros(85))
# for order in 1:85
#     @load expres_save_path * star * "/$(order)/data.jld2" used_excal
#     excals[order] = used_excal
# end
# findfirst(excals):findlast(excals)
excalibur_orders_list = [42:77, 40:77, 38:77]

# for star_ind in 1:3
#     for prep_str in ["wobble_", ""]
star = stars[star_ind]
orders = orders_list[star_ind]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]

## RV reduction

@load "jld2/expres_$(prep_str)$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
# rvs = rvs[orders, :]  # TODO: make this more robust to different starting order numbers
# rvs_σ = rvs_σ[orders, :]  # TODO: make this more robust to different starting order numbers
# # plotting order means which don't matter because the are constant shifts for the reduced rv
# _scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= median(rvs; dims=2)
χ² = vec(sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2))
med_rvs_σ = vec(median(rvs_σ; dims=2))
rvs_std = vec(std(rvs; dims=2))
σ_floor = 50

χ²_sortperm = [i for i in sortperm(χ²) if !iszero(χ²[i])]
χ²_thres = 1e3 / 114 * size(rvs, 2)
χ²_orders = [i for i in χ²_sortperm if χ²[i] < χ²_thres]
χ²_orders = [orders[χ²_order] for χ²_order in χ²_orders]
inds = orders2inds([orders[i] for i in eachindex(orders) if ((rvs_std[i] < σ_floor) && (med_rvs_σ[i] < σ_floor) && (orders[i] in χ²_orders) && (!only_excalibur || (orders[i] in excalibur_orders_list[star_ind])))])
println("starting with $(length(orders)) orders")
println("$(length(orders) - length(χ²_sortperm)) orders ignored for unfinished analyses or otherwise weird")
println("$(length(χ²_sortperm) - length(χ²_orders)) worst orders (in χ²-sense) ignored")
only_excalibur ?
    println("$(length(χ²_orders) - length(inds)) more orders ignored for having >$σ_floor m/s RMS or errors or aren't excalibur orders") :
    println("$(length(χ²_orders) - length(inds)) more orders ignored for having >$σ_floor m/s RMS or errors")
println("$(length(inds)) orders used in total")

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_red .-= median(rvs_red)
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
expres_rv = expres_output."CBC RV [m/s]"
expres_rv .-= median(expres_rv)
expres_rv_σ = expres_output."CBC RV Err. [m/s]"
expres_time = expres_output."Time [MJD]"
mask = Bool.(ones(length(times_nu)))

using LinearAlgebra
lin = SSOF.general_lst_sq_f(rvs_red, Diagonal(rvs_σ2_red), 1; x=times_nu)

annot = text.(orders[rvs_std .< σ_floor], :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders[rvs_std .< σ_floor], rvs_std[rvs_std .< σ_floor]; legend=:topleft, label="", title="$star RV std", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), ylim=[0,σ_floor], series_annotations=annot, markerstrokewidth=0.5)
annot = text.(orders[rvs_std .> σ_floor], :center, :black, 3)
scatter!(plt, orders[rvs_std .> σ_floor], ones(sum(rvs_std .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1], markerstrokewidth=0.5)
png(plt, "figs/expres_" * prep_str * star * "_order_rv_std")

annot = text.(orders[med_rvs_σ .< σ_floor], :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders[med_rvs_σ .< σ_floor], med_rvs_σ[med_rvs_σ .< σ_floor]; legend=:topleft, label="", title="$star Median σ", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor], markerstrokewidth=0.5)
annot = text.(orders[med_rvs_σ .> σ_floor], :center, :black, 3)
scatter!(plt, orders[med_rvs_σ .> σ_floor], ones(sum(med_rvs_σ .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1], markerstrokewidth=0.5)
png(plt, "figs/expres_" * prep_str * star * "_order_rv_σ")

annot = text.(orders, :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders, std(rvs; dims=2) ./ med_rvs_σ; legend=:topleft, label="", title="$star (RV std) / (Median σ)", xlabel="Order", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, markerstrokewidth=0.5)
png(plt, "figs/expres_" * prep_str * star * "_order_rv_ratio")

annot = text.(orders[χ²_sortperm], :center, :black, 4)
plt = SSOFU._plot()
scatter!(plt, χ²[χ²_sortperm]; label="", ylabel="χ²", series_annotations=annot, title = star * "_χ²", size=(SSOFU._plt_size[1]*(length(orders) / 80),SSOFU._plt_size[2]), markerstrokewidth=0, yaxis=:log10)
hline!(plt, [χ²_thres]; label="", lw=1, color=SSOFU.plt_colors[1])
png(plt, "figs/expres_" * prep_str * star * "_χ²")

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu[mask], rvs_red[mask], rvs_σ_red[mask], expres_time[mask], expres_rv[mask], expres_rv_σ[mask]; markerstrokewidth=0.5, title="HD $star (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, "figs/expres_" * prep_str * star * "_model_rvs.png")
plt = SSOFU.plot_model_rvs(times_nu[mask], rvs_red[mask] .- lin.(times_nu[mask]), rvs_σ_red[mask], expres_time[mask], expres_rv[mask], expres_rv_σ[mask]; markerstrokewidth=0.5, title="HD $star (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, "figs/expres_" * prep_str * star * "_model_rvs_detrend.png")
#     end
# end
