## Importing packages
using Pkg
Pkg.activate("EXPRES")

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using Plots
import StatsBase

## Setting up necessary variables

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 2)]
interactive = length(ARGS) == 0
save_plots = true
include("data_locs.jl")  # defines expres_data_path and expres_save_path
desired_order = SSOF.parse_args(2, Int, 68)  # 68 has a bunch of tels, 47 has very few
use_reg = SSOF.parse_args(3, Bool, true)

## Loading in data and initializing model
save_path = expres_save_path * star * "/$(desired_order)/"
@load save_path * "data.jld2" n_obs data times_nu airmasses
if !use_reg
    save_path *= "noreg_"
end

@time test_model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=2, n_comp_star=2)
@time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(test_model, data; use_gp=true)
if !use_reg
    SSOF.zero_regularization(test_model)
    test_model.metadata[:todo][:reg_improved] = true
end

if isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" model
    SSOF.copy_reg!(model, test_model)
    test_model.metadata[:todo][:reg_improved] = true
end

## Creating optimization workspace
workspace = SSOF.OptimWorkspace(test_model, data)
workspace = SSOF.TotalWorkspace(test_model, data)
workspace = SSOF.TelStarWorkspace(test_model, data)

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(workspace)
else
    ENV["GKSwstype"] = "100"  # setting the GR workstation type to 100/nul
end

## Optimizing model

@time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
rvs_notel_opt = SSOF.rvs(test_model)
if interactive; status_plot(workspace.o, workspace.d) end
test_model.metadata[:todo][:optimized] = true

## Testing different models

data.var[data.var.==Inf] .= 0
data_noise = sqrt.(data.var)
data.var[data.var.==0] .= Inf
data_holder = copy(data)
n_err = 25
n_obs = length(test_model.rv.lm.s)
rv_holder = Array{Float64}(undef, n_err, n_obs)

test_n_comp_tel = 0:2
test_n_comp_star = 0:2
rvs = Array{Float64}(undef, length(test_n_comp_tel), length(test_n_comp_star), n_obs)
rvs_σ = Array{Float64}(undef, length(test_n_comp_tel), length(test_n_comp_star), n_obs)
for (i, n_tel) in enumerate(test_n_comp_tel)
    for (j, n_star) in enumerate(test_n_comp_star)
        ws = SSOF.OptimWorkspace(SSOF.downsize(test_model, n_tel, n_star), data)
        SSOF.fine_train_OrderModel!(ws)
        rvs[i, j, :] .= SSOF.rvs(ws.om)
        model_holder = copy(ws.om)
        @time for i in 1:n_err
            data_holder.flux .= data.flux + (data_noise .* randn(size(data_holder.var)))
            SSOF.train_OrderModel!(SSOF.OptimWorkspace(model_holder, data_holder), f_tol=1e-8)
            rv_holder[i, :] .= SSOF.rvs(model_holder)
        end
        rvs_σ[i, j, :] .= vec(std(rv_holder; dims=1))
        @save save_path*"low_comp_rvs.jld2" rvs rvs_σ test_n_comp_tel test_n_comp_star
    end
end

@save save_path*"low_comp_rvs.jld2" rvs rvs_σ test_n_comp_tel test_n_comp_star
