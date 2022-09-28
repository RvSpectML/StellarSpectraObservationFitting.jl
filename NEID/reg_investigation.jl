## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
using JLD2
using Statistics
import StatsBase
using Printf

## Setting up necessary variables

stars = ["10700"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 60)  # 81 has a bunch of tels, 60 has very few
max_components = 8

## Loading in data and initializing model and initial training
save_path = neid_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses

@time model = SSOF.OrderModel(data, "NEID", desired_order, star; n_comp_tel=max_components, n_comp_star=max_components)
@time rvs_notel, fracvar_tel, fracvar_star = SSOF.initialize!(model, data)
mws = SSOF.OptimWorkspace(model, data)
@time SSOF.train_OrderModel!(mws; verbose=true, ignore_regularization=true)  # 45s

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/src/_plot_functions.jl")
if !interactive; ENV["GKSwstype"] = "100" end

## Initial model selection

test_n_comp_tel = 0:max_components
test_n_comp_star = 0:max_components
ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
for (i, n_tel) in enumerate(test_n_comp_tel)
    for (j, n_star) in enumerate(test_n_comp_star)
        _mws = typeof(mws)(SSOF.downsize(mws.om, n_tel, n_star), mws.d)
        comp_ls[i, j], ks[i, j] = SSOF._loss(_mws), SSOF.total_length(_mws)
    end
end
n_comps_best, ℓ, aics, bics = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, data.var; return_inters=true)

# downsizing and refitting
model = SSOF.downsize(model, n_comps_best[1]+1, n_comps_best[2]+1)
mws = typeof(mws)(model, data)
@time SSOF.train_OrderModel!(mws; verbose=true, ignore_regularization=true)  # 45s

# plt = status_plot(mws)
# png(plt, "test7")
# plt = plot_model(mws)
# png(plt, "test8")
## Regularization investigation

n_obs_test = Int(round(0.25 * n_obs))
test_start_ind = Int(round(rand() * (n_obs - n_obs_test)))
testing_inds = test_start_ind:test_start_ind+n_obs_test-1
training_inds = [i for i in 1:n_obs if !(i in testing_inds)]
SSOF.zero_regularization(model)

regs = [10.0^i for i in 2:12]
regs[1] = 0
losses = zeros(length(regs))

function test_reg(reg_sym::Symbol, tel::Bool)
    if tel
        dic = mws.om.reg_tel
        s = "tel"
    else
        dic = mws.om.reg_star
        s = "star"
    end
    reg_s = string(reg_sym)
    for i in 1:length(regs)
        reg = regs[i]
        om, d = copy(mws.om), mws.d
        dic[reg_sym] = reg
        train = typeof(mws)(om, d, training_inds)
        test = typeof(mws)(om, d, testing_inds; only_s=true)
        SSOF.train_OrderModel!(train) # trains basis vectors and (scores at training time)
        SSOF.train_OrderModel!(test)  # trains scores at testing times
        losses[i] = SSOF._loss(test)
    end
    dic[reg_sym] = 0
    scatter(regs[2:end], losses[2:end]; xaxis=:log, yaxis=:log, label="")
    hline!([losses[1]], label="no reg")
    @save "jld2/neid/"*star*"_$(desired_order)_"*reg_s*"_"*s*".jld2" losses regs
end

reg_syms =  [:GP_μ, :L1_μ, :L1_M, :GP_M]
for reg_sym in reg_syms
    @time test_reg(reg_sym, true)
    println("done with " * string(reg_sym) * " tel")
    @time test_reg(reg_sym, false)
    println("done with " * string(reg_sym) * " star")
end

plt = _plot(; title="NEID HD " * star * " Order " * string(desired_order), legend=:topleft, xaxis=:log, yaxis=:log, ylabel="χ² of training set", xlabel="regularization value")
for (i, s) in enumerate(["tel", "star"])
    for (j, reg_s) in enumerate(string.(reg_syms))
        c = plt_colors[c_ind_f(length(reg_syms)*(i-1)+j)]
        @load "jld2/neid/"*star*"_$(desired_order)_"*reg_s*"_"*s*".jld2" losses regs
        scatter!(plt, regs[2:end], losses[2:end]; label="", markerstrokewidth=0, color=c, alpha=0.3)
        min_i = max(2,argmin(losses))
        # vline!(plt, [regs[max(2,argmin(losses))]], label="", color=plt_colors[c_ind_f(4*i+j)])
        scatter!(plt, [regs[min_i]], [losses[min_i]]; label=s*" "*reg_s * (@sprintf " %.0E" regs[min_i]) * " (Δχ²=$(round(1 - losses[min_i]/losses[1], digits=3)))", markerstrokewidth=0, markersize=6, markershape=:x, color=c)
        hline!(plt, [losses[min_i]], label="", color=c, alpha=0.3)
    end
end
@load "jld2/neid/"*star*"_$(desired_order)_GP_μ_tel.jld2" losses regs
hline!(plt, [losses[1]], label="baseline")
ylims!(plt, 10^5, 10^6)
png(plt, "figs/NEID_" * star * "_$(desired_order)_reg_investigation")

# plot_stars = ["expres/26965", "expres/34411", "expres/10700", "neid/10700"]
# Δχ² = zeros(2, 4, 2, length(reg_syms))
# regs_all = zeros(Int, 2, 4, 2, length(reg_syms))
# for (l, order) in enumerate([47, 68])
#     for (k, star) in enumerate(plot_stars)
#         if k > 3; order += 13 end
#         for (i, s) in enumerate(["tel", "star"])
#             for (j, reg_s) in enumerate(string.(reg_syms))
#                 @load "jld2/"*star*"_$(order)_"*reg_s*"_"*s*".jld2" losses regs
#                 min_i = max(2,argmin(losses))
#                 regs_all[l, k, i, j] = Int(log10(regs[min_i]))
#                 Δχ²[l, k, i, j] = 1 - losses[min_i]/losses[1]
#             end
#         end
#     end
# end
#
# plot_i = 1
# for plot_i in 1:2
#     title = ["no tels", "some tels"][plot_i]
#     ind_f(x) = view(x, plot_i, :, 1, :)
#     heatmap(string.(reg_syms), plot_stars, ind_f(Δχ²); annotationcolor=:white, title=title)
#     annotate!(vec(tuple.((1:length(reg_syms))'.-0.5, (1:length(plot_stars)).-0.5, string.(round.(ind_f(Δχ²), digits=3)))))
#     png("heatmap_$plot_i")
#     heatmap(string.(reg_syms), plot_stars, ind_f(regs_all); annotationcolor=:white, title=title)
#     annotate!(vec(tuple.((1:length(reg_syms))'.-0.5, (1:length(plot_stars)).-0.5, ind_f(regs_all))))
#     png("regs_$plot_i")
# end
