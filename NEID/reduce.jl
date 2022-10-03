## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using CSV, DataFrames
using Statistics
using Plots
using LinearAlgebra

## Setting up necessary variables

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
stars = ["10700", "26965", "22049", "3651", "95735", "2021/12/19", "2021/12/20", "2021/12/23"]
orders_list = repeat([7:118], length(stars))
include("data_locs.jl")  # defines neid_data_path and neid_save_path
star_ind = SSOF.parse_args(1, Int, 2)
bootstrap = SSOF.parse_args(2, Bool, false)
log_lm = SSOF.parse_args(3, Bool, true)
dpca = SSOF.parse_args(4, Bool, false)
use_lsf = SSOF.parse_args(5, Bool, false)

bootstrap ? appe_str = "_boot" : appe_str = "_curv"
log_lm ? prep_str = "log_" : prep_str = "lin_"
dpca ? prep_str *= "dcp_" : prep_str *= "vil_"
use_lsf ? prep_str *= "lsf_" : prep_str *= "nol_"

# for star_ind in [1,2,5]
#     for prep_str in ["wobble_", ""]
star = stars[star_ind]
save_fn = "figs/neid_" * prep_str * star * appe_str * "_"
orders = orders_list[star_ind]
orders2inds(selected_orders::AbstractVector) = [searchsortedfirst(orders, order) for order in selected_orders]

## RV reduction
@load "jld2/neid_$(prep_str)$(star)_rvs$appe_str.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

@load neid_save_path * star * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ
neid_rv .-= median(neid_rv)
star=="10700" ? mask = .!(2459525 .< times_nu .< 2459530) : mask = Bool.(ones(length(times_nu)))

# # plotting order means which don't matter because the are constant shifts for the reduced rv
# _scatter(orders, mean(rvs; dims=2); series_annotations=annot, legend=:topleft)
rvs .-= median(rvs; dims=2)
χ² = vec(sum((rvs .- mean(rvs; dims=2)) .^ 2 ./ (rvs_σ .^ 2); dims=2))
med_rvs_σ = vec(median(rvs_σ; dims=2))
rvs_std = vec(std(rvs; dims=2))
std_floor = round(8 * std(neid_rv[mask]))
σ_floor = std_floor / 3

χ²_sortperm = [i for i in sortperm(χ²) if !iszero(χ²[i])]
χ²_thres = 4.5e3 / 37 * size(rvs, 2)
χ²_orders = [i for i in χ²_sortperm if χ²[i] < χ²_thres]
χ²_orders = [orders[χ²_order] for χ²_order in χ²_orders]
inds = orders2inds([orders[i] for i in eachindex(orders) if ((rvs_std[i] < std_floor) && (med_rvs_σ[i] < σ_floor) && (orders[i] in χ²_orders))])
inds = orders2inds([orders[i] for i in eachindex(orders) if ((rvs_std[i] < std_floor) && (med_rvs_σ[i] < σ_floor) && true)])
println("starting with $(length(orders)) orders")
println("$(length(orders) - length(χ²_sortperm)) orders ignored for unfinished analyses or otherwise weird")
println("$(length(χ²_sortperm) - length(χ²_orders)) worst orders (in χ²-sense) ignored")
println("$(length(χ²_orders) - length(inds)) more orders ignored for having >$std_floor m/s RMS or >$σ_floor errors")
println("$(length(inds)) orders used in total")

rvs_red = collect(Iterators.flatten((sum(rvs[inds, :] ./ (rvs_σ[inds, :] .^ 2); dims=1) ./ sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1))'))
rvs_red .-= median(rvs_red)
rvs_σ_red = collect(Iterators.flatten(1 ./ sqrt.(sum(1 ./ (rvs_σ[inds, :] .^ 2); dims=1)')))

using LinearAlgebra
lin = SSOF.general_lst_sq_f(rvs_red, Diagonal(rvs_σ_red .^ 2), 1; x=times_nu)

annot = text.(orders[rvs_std .< std_floor], :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders[rvs_std .< std_floor], rvs_std[rvs_std .< std_floor]; legend=:topleft, label="", title="$star RV std", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), ylim=[0, std_floor], series_annotations=annot, markerstrokewidth=0.5)
annot = text.(orders[rvs_std .> std_floor], :center, :black, 3)
scatter!(plt, orders[rvs_std .> std_floor], ones(sum(rvs_std .> std_floor)) .* std_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1], markerstrokewidth=0.5)
png(plt, save_fn * "order_rv_std")

annot = text.(orders[med_rvs_σ .< σ_floor], :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders[med_rvs_σ .< σ_floor], med_rvs_σ[med_rvs_σ .< σ_floor]; legend=:topleft, label="", title="$star Median σ", xlabel="Order", ylabel="m/s", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, ylim=[0,σ_floor], markerstrokewidth=0.5)
annot = text.(orders[med_rvs_σ .> σ_floor], :center, :black, 3)
scatter!(plt, orders[med_rvs_σ .> σ_floor], ones(sum(med_rvs_σ .> σ_floor)) .* σ_floor; label="", series_annotations=annot, markershape=:utriangle, c=SSOFU.plt_colors[1], markerstrokewidth=0.5)
png(plt, save_fn * "order_rv_σ")

annot = text.(orders, :center, :black, 3)
plt = SSOFU._plot()
scatter!(plt, orders, std(rvs; dims=2) ./ med_rvs_σ; legend=:topleft, label="", title="$star (RV std) / (Median σ)", xlabel="Order", size=(SSOFU._plt_size[1]*0.5,SSOFU._plt_size[2]*0.75), series_annotations=annot, markerstrokewidth=0.5)
png(plt, save_fn * "order_rv_ratio")

annot = text.(orders[χ²_sortperm], :center, :black, 4)
plt = SSOFU._plot(; ylabel="χ²", title = star * "_χ²", size=(SSOFU._plt_size[1]*(length(orders) / 80),SSOFU._plt_size[2]), yaxis=:log10)
scatter!(plt, χ²[χ²_sortperm]; label="", series_annotations=annot, markerstrokewidth=0)
hline!(plt, [χ²_thres]; label="", lw=1, color=SSOFU.plt_colors[1])
png(plt, save_fn * "χ²")

# Compare RV differences to actual RVs from activity
plt = SSOFU.plot_model_rvs(times_nu[mask], rvs_red[mask], rvs_σ_red[mask], neid_time[mask], neid_rv[mask], neid_rv_σ[mask]; markerstrokewidth=0.5, title="HD $star (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, save_fn * "model_rvs.png")
plt = SSOFU.plot_model_rvs(times_nu, rvs_red .- lin.(times_nu), rvs_σ_red, neid_time, neid_rv, neid_rv_σ; markerstrokewidth=0.5, title="HD $star (median σ: $(round(median(rvs_σ_red), digits=3)))")
png(plt, save_fn * "model_rvs_detrend.png")
#     end
# end

# RV differnces as a function of order (annot somehow breaks everything?)
x = rvs[inds, :] .- rvs_red'
x_σ = vec(median(rvs_σ[inds, :]; dims=2))
# annot = text.(orders[inds], :center, :black, 3)
plt = SSOFU._plot(; xlabel="Order", ylabel="std(RV_order - RV_bulk) (m/s)", title = star, size=(SSOFU._plt_size[1]*(length(orders) / 80),SSOFU._plt_size[2]))
scatter!(plt, orders[inds], vec(std(x; dims=2)); yerror=x_σ, label="", markerstrokewidth=0.5)#, series_annotations=annot)
png(plt, save_fn * "order_dif")

# order rv correlation heatmap
n_ord = size(rvs, 1)
cors = Array{Float64}(undef, n_ord, n_ord)
for i in 1:n_ord
    for j in i+1:n_ord
        cors[i, j] = cor(view(rvs, i, :), view(rvs, j, :))
    end
end
cors[diagind(cors)] .= 1
cors = Symmetric(cors)
heatmap(Matrix(cors))

plt = SSOFU._plot(; xlabel="Order", title="Correlation with Bulk RVs (error bars ∝ σᵣᵥ)", ylabel="Correlation")
scatter!(orders, [cor(view(rvs, i, :), rvs_red) for i in 1:n_ord]; yerror=x_σ./20, ylims=(-0.5, 1.1), markerstrokewidth=0.5, label="Unused Orders")
scatter!(orders[inds], [cor(view(rvs, i, :), rvs_red) for i in 1:n_ord][inds]; label="Used Orders", markerstrokewidth=0.5)
png(plt, save_fn * "order_cor")
