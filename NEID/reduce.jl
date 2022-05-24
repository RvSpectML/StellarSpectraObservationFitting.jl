## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using CSV, DataFrames

## Setting up necessary variables

SSOF_path = dirname(dirname(pathof(SSOF)))
stars = ["10700", "26965", "9407", "185144", "2021/12/19", "2021/12/20", "2021/12/23"]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]
# prep_str = "wobble_"
prep_str = ""

# for star_ind in 1:2
star_ind = SSOF.parse_args(1, Int, 2)
orders_list = repeat([4:122], length(stars))
star = stars[star_ind]
orders = orders_list[star_ind]

## Looking at model components

@load "jld2/neid_$(prep_str)$(star)_md.jld2" n_comps n_comps_bic robust

n_robust = .!robust
x = orders_list[star_ind]
annot=text.(x, :top, :white, 5)
α = 1
# robust_str = ["" for i in x]
# for i in eachindex(robust_str)
#     robust_str[i] *= "$(x[i])"
#     if !robust[i]; robust_str[i] *= "!" end
# end
# annot=text.(robust_str, :top, :white, 9)
plt = _plot(; ylabel="# of basis vectors", xlabel="Order", title="Best Models for $star (Based on AIC)", xticks=false)
_scatter!(plt, x, n_comps[:, 1]; alpha=α, label="# of telluric components", legend=:top, series_annotations=annot)
_scatter!(plt, x, n_comps[:, 2]; alpha=α, label="# of stellar components", series_annotations=annot)
plot!(plt, x, n_comps[:, 1]; label = "", alpha=α, color=plt_colors[1], ls=:dot)
plot!(plt, x, n_comps[:, 2]; label = "", alpha=α, color=plt_colors[2], ls=:dot)
_scatter!(plt, x[n_robust], n_comps_bic[n_robust, 1]; alpha=α/2, color=plt_colors[11], label="# of telluric components (BIC)")
_scatter!(plt, x[n_robust], n_comps_bic[n_robust, 2]; alpha=α/2, color=plt_colors[12], label="# of stellar components (BIC)")
plot!(plt, x, n_comps_bic[:, 1]; label = "", alpha=α/2, color=plt_colors[11], ls=:dot)
plot!(plt, x, n_comps_bic[:, 2]; label = "", alpha=α/2, color=plt_colors[12], ls=:dot)

png(plt, "neid_$(prep_str)md_$star.png")

## RV reduction

@load "jld2/neid_$(prep_str)$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
# # plotting order means which don't matter because the are constant shifts for the reduced rv
# _scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= median(rvs; dims=2)
χ² = vec(sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2))
med_rvs_σ = vec(median(rvs_σ; dims=2))
rvs_std = vec(std(rvs; dims=2))
σ_floor = 50

χ²_orders = sortperm(χ²)[1:end-20]
χ²_orders = [orders[χ²_order] for χ²_order in χ²_orders]
inds = orders2inds([orders[i] for i in eachindex(orders) if (med_rvs_σ[i] < σ_floor) && (orders[i] in χ²_orders)])

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_red .-= median(rvs_red)
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))
rvs_σ2_red = rvs_σ_red .^ 2

@load SSOF_path * "/NEID/" * star * "_neid_pipeline.jld2" neid_time neid_rv neid_rv_σ
star=="10700" ? mask = .!(2459525 .< times_nu .< 2459530) : mask = Bool.(ones(length(times_nu)))

neid_rv .-= median(neid_rv)

using LinearAlgebra
lin = SSOF.general_lst_sq_f(rvs_red, Diagonal(rvs_σ2_red), 1; x=times_nu)


annot = text.(orders[rvs_std .< σ_floor], :center, :black, 3)
plt = SSOFU._scatter(orders[rvs_std .< σ_floor], rvs_std[rvs_std .< σ_floor]; legend=:topleft, label="", title="$star RV std", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor])
annot = text.(orders[rvs_std .> σ_floor], :center, :black, 3)
SSOFU._scatter!(plt, orders[rvs_std .> σ_floor], ones(sum(rvs_std .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1])
png(plt, "neid_" * prep_str * star * "_order_rv_std")

annot = text.(orders[med_rvs_σ .< σ_floor], :center, :black, 3)
plt = SSOFU._scatter(orders[med_rvs_σ .< σ_floor], med_rvs_σ[med_rvs_σ .< σ_floor]; legend=:topleft, label="", title="$star Median σ", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor])
annot = text.(orders[med_rvs_σ .> σ_floor], :center, :black, 3)
SSOFU._scatter!(plt, orders[med_rvs_σ .> σ_floor], ones(sum(med_rvs_σ .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1])
png(plt, "neid_" * prep_str * star * "_order_rv_σ")

annot = text.(orders, :center, :black, 3)
plt = SSOFU._scatter(orders, std(rvs; dims=2) ./ med_rvs_σ; legend=:topleft, label="", title="$star (RV std) / (Median σ)", xlabel="Order", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot)
png(plt, "neid_" * prep_str * star * "_order_rv_ratio")

annot = text.(orders[sortperm(χ²)], :center, :black, 4)
plt = SSOFU._scatter(1:length(χ²), sort(χ²); label="χ²", series_annotations=annot, legend=:topleft, title=prep_str * star * "_χ²", size=(SSOFU._plt_size[1]*(length(orders) / 80),SSOFU._plt_size[2]), markerstrokewidth=0) #, yaxis=:log)
png(plt, "neid_" * prep_str * star * "_χ²")

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu[mask], rvs_red[mask], rvs_σ_red[mask], neid_time[mask], neid_rv[mask], neid_rv_σ[mask]; markerstrokewidth=1, title="HD $star (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, "neid_" * prep_str * star * "_model_rvs.png")
plt = SSOFU.plot_model_rvs(times_nu[mask], rvs_red[mask] .- lin.(times_nu), rvs_σ_red[mask], neid_time[mask], neid_rv[mask], neid_rv_σ[mask]; markerstrokewidth=1, title="Sun 12/19/2021 (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, "neid_" * prep_str * star * "_model_rvs_detrend.png")
# end

## regularization by order

@load "neid_$(prep_str)$(star)_regs.jld2" reg_tels reg_stars
reg_keys = SSOF._key_list[1:end-1]
mask = [reg_tels[i, 1]!=0 for i in 1:length(orders)]

plt = _plot(;xlabel="Order", ylabel="Regularization", title="Regularizations per order (HD$star)", yaxis=:log)
for i in eachindex(reg_keys)
    plot!(plt, orders[mask], reg_tels[mask, i], label="reg_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
end
# for i in eachindex(reg_keys)
#     plot!(plt, orders, reg_stars[:, i], label="star_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
# end
for i in eachindex(reg_keys)
    hline!(plt, [SSOF.default_reg_tel[reg_keys[i]]], c=plt_colors[i], label="")
    # hline!(plt, [SSOF.default_reg_star[reg_keys[i]]], c=plt_colors[i+length(reg_keys)], label="")
end
display(plt)
png(plt, "neid_" * prep_str * star * "_reg_tel")

plt = _plot(;xlabel="Order", ylabel="Regularization", title="Regularizations per order (HD$star)", yaxis=:log)
# for i in eachindex(reg_keys)
#     plot!(plt, orders, reg_tels[:, i], label="reg_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
# end
for i in eachindex(reg_keys)
    plot!(plt, orders[mask], reg_stars[mask, i], label="star_$(reg_keys[i])", markershape=:circle, markerstrokewidth=0)
end
for i in eachindex(reg_keys)
    # hline!(plt, [SSOF.default_reg_tel[reg_keys[i]]], c=plt_colors[i], label="")
    hline!(plt, [SSOF.default_reg_star[reg_keys[i]]], c=plt_colors[i], label="")
end
display(plt)
png(plt, "neid_" * prep_str * star * "_reg_star")
