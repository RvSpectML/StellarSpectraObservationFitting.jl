## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using CSV, DataFrames

## Setting up necessary variables

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
dates = ["2021/12/10", "2021/12/19", "2021/12/20", "2021/12/23"]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]
# prep_str = "noreg_"
prep_str = "nolsf_"

# for date_ind in 1:2
date_ind = SSOF.parse_args(1, Int, 3)
orders_list = [7:118, 7:118, 7:118, 7:118]
date = dates[date_ind]
date_ = replace(date, "/" => "_")
orders = orders_list[date_ind]

include("data_locs.jl")  # defines neid_data_path and neid_save_path

## RV reduction
@load neid_save_path*date*"/$(prep_str)rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
# # plotting order means which don't matter because the are constant shifts for the reduced rv
# _scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= median(rvs; dims=2)
med_rvs_σ = vec(median(rvs_σ; dims=2))
rvs_std = vec(std(rvs; dims=2))
σ_floor = 50

annot = text.(orders[rvs_std .< σ_floor], :center, :black, 3)
plt = _scatter(orders[rvs_std .< σ_floor], rvs_std[rvs_std .< σ_floor]; legend=:topleft, label="", title="$date RV std", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor])
annot = text.(orders[rvs_std .> σ_floor], :center, :black, 3)
_scatter!(plt, orders[rvs_std .> σ_floor], ones(sum(rvs_std .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=plt_colors[1])
png(plt, "neid_" * prep_str * date_ * "_order_rv_std")

annot = text.(orders[med_rvs_σ .< σ_floor], :center, :black, 3)
plt = _scatter(orders[med_rvs_σ .< σ_floor], med_rvs_σ[med_rvs_σ .< σ_floor]; legend=:topleft, label="", title="$date Median σ", xlabel="Order", ylabel="m/s", size=(_plt_size[1]*0.5,_plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor])
annot = text.(orders[med_rvs_σ .> σ_floor], :center, :black, 3)
_scatter!(plt, orders[med_rvs_σ .> σ_floor], ones(sum(med_rvs_σ .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=plt_colors[1])
png(plt, "neid_" * prep_str * date_ * "_order_rv_σ")

annot = text.(orders, :center, :black, 3)
plt = _scatter(orders, std(rvs; dims=2) ./ med_rvs_σ; legend=:topleft, label="", title="$date (RV std) / (Median σ)", xlabel="Order", size=(_plt_size[1]*0.5,_plt_size[2]*0.75), series_annotations=annot)
png(plt, "neid_" * prep_str * date_ * "_order_rv_ratio")

χ² = vec(sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2))
annot = text.(orders[sortperm(χ²)], :center, :black, 4)

plt = _scatter(1:length(χ²)-1, sort(χ²)[1:end-1]; label="χ²", series_annotations=annot, legend=:topleft, title=prep_str * date * "_χ²") #, yaxis=:log)
png(plt, "neid_" * prep_str * date_ * "_χ²")

χ²_orders = sortperm(χ²)[1:end-20]
χ²_orders = [orders[χ²_order] for χ²_order in χ²_orders]
inds = orders2inds([orders[i] for i in eachindex(orders) if (med_rvs_σ[i] < σ_floor) && (orders[i] in χ²_orders)])

orders[inds]

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_red .-= median(rvs_red)
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

@load neid_save_path*date*"/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ
neid_rv .-= median(neid_rv)

# Compare RV differences to actual RVs from activity
plt = plot_model_rvs(times_nu, rvs_red, rvs_σ_red, neid_time, neid_rv, neid_rv_σ; markerstrokewidth=1, title="$date (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, "neid_" * prep_str * date_ * "_model_rvs_mask.png")
# end

## regularization by order

@load neid_save_path*date*"/$(prep_str)regs.jld2" reg_tels reg_stars
reg_keys = [:GP_μ, :L1_μ, :L1_μ₊_factor, :GP_M, :L1_M]
mask = [reg_tels[i, 1]!=0 for i in 1:length(orders)]

plt = _plot(;xlabel="Order", ylabel="Regularization", title="Regularizations per order ($date)", yaxis=:log)
for i in eachindex(reg_keys)
    plot!(plt, orders[mask], reg_tels[mask, i], label="reg_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
end
# for i in eachindex(reg_keys)
#     plot!(plt, orders, reg_stars[:, i], label="date_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
# end
for i in eachindex(reg_keys)
    hline!(plt, [SSOF.default_reg_tel[reg_keys[i]]], c=plt_colors[i], label="")
    # hline!(plt, [SSOF.default_reg_star[reg_keys[i]]], c=plt_colors[i+length(reg_keys)], label="")
end
display(plt)
png(plt, "neid_" * prep_str * date_ * "_reg_tel")

plt = _plot(;xlabel="Order", ylabel="Regularization", title="Regularizations per order ($date)", yaxis=:log)
# for i in eachindex(reg_keys)
#     plot!(plt, orders, reg_tels[:, i], label="reg_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
# end
for i in eachindex(reg_keys)
    plot!(plt, orders[mask], reg_stars[mask, i], label="date_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
end
for i in eachindex(reg_keys)
    # hline!(plt, [SSOF.default_reg_tel[reg_keys[i]]], c=plt_colors[i], label="")
    hline!(plt, [SSOF.default_reg_star[reg_keys[i]]], c=plt_colors[i], label="")
end
display(plt)
png(plt, "neid_" * prep_str * date_ * "_reg_star")
