## Importing packages
using Pkg
Pkg.activate("EXPRES")

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
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

if isfile(save_path*"results.jld2")
    @load save_path*"results.jld2" model rvs_naive rvs_notel
    if model.metadata[:todo][:err_estimated]
        @load save_path*"results.jld2" rv_errors
    end
    if model.metadata[:todo][:downsized]
        @load save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    end
else
    model_upscale = sqrt(2)
    # model_upscale = 2 * sqrt(2)
    @time model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8, upscale=model_upscale)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data; use_gp=true)
    if !use_reg
        SSOF.zero_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end


## Creating optimization workspace
workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)

ts = workspace.telstar
using BenchmarkTools
using Zygote
using ParameterHandling
using BandedMatrices
@btime ts.obj.f(ts.p0)
Zygote.gradient(ts.obj.f, ts.p0)

x = 0.5:0.5:1000
xo = 1.25:1:999
xob = SSOF.bounds_generator(xo)
θ = ones(2000,3) .+ (0.005 .* x)
y = randn(3,100)

θ*y
bm = BandedMatrix(ones(1000,1000),(13,13))
function f(x)
    ans = θ*y
    return mapreduce(i -> sum(bm * view(ans, :, i)), +, 1:size(ans, 2))
end
xf, unflatten = flatten(x)
fz = f ∘ unflatten
Zygote.gradient(fz, xf)
Zygote.pullback(fz, xf)

SSOF.oversamp_interp(lo_x::Real, hi_x::Real, x::AbstractVector, y::AbstractVector)
[SSOF.oversamp_interp(xob[i], xob[i+1], x, θ[:, 1]) for i in 1:(length(xob)-1)]


θ
xobs = repeat(xob, 1, 100)
vals = θ*y
@time SSOF.spectra_interp(vals, x, xobs)
@time SSOF.spectra_interp_old(vals, x, xobs)

function spectra_interp_old(vals::AbstractMatrix, basis::AbstractVector, bounds::AbstractMatrix)
	interped_vals = zeros(size(bounds, 1)-1, size(bounds, 2))
	for i in 1:size(interped_vals, 2)
		interped_vals[:, i] .= spectra_interp(view(vals, :, i), basis, view(bounds, :, i))
	end
	return interped_vals
end
spectra_interp(vals::AbstractMatrix, basis::AbstractVector, bounds::AbstractMatrix) =
	hcat([spectra_interp(view(vals, :, i), basis, view(bounds, :, i)) for i in 1:size(bounds, 2)]...)

## Plotting

SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
    status_plot(workspace.o, workspace.d)
else
    ENV["GKSwstype"] = "100"  # setting the GR workstation type to 100/nul
end

## Improving regularization

if false#!model.metadata[:todo][:reg_improved]
    @time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true, ignore_regularization=true)  # 16s
    n_obs_train = Int(round(0.75 * n_obs))
    training_inds = sort(StatsBase.sample(1:n_obs, n_obs_train; replace=false))
    @time SSOF.fit_regularization!(model, data, training_inds)
    model.metadata[:todo][:reg_improved] = true
    model.metadata[:todo][:optimized] = false
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end

## Optimizing model

if !model.metadata[:todo][:optimized]
    @time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
    rvs_notel_opt = SSOF.rvs(model)
    if interactive; status_plot(workspace.o, workspace.d) end
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel
end
status_plot(workspace.o, workspace.d)

# @time results_telstar, _ = SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
ts = workspace.telstar
@time ts.obj.f(ts.p0)
Zygote.gradient(ts.obj.f, ts.p0)
ts.obj.df(g, ts.p0)

using Optim
ow = workspace; print_stuff=true; iterations=1; f_tol=1e-6; g_tol=2.5e5;
optim_cb_local(x::OptimizationState) = SSOF.optim_cb(x; print_stuff=print_stuff)

options = Optim.Options(;iterations=iterations, f_tol=f_tol, g_tol=g_tol, callback=optim_cb_local)


# optimize tellurics and star
result_telstar, nt = SSOF._OSW_optimize!(ow.telstar, options)

if ow.only_s
    SSOF._custom_copy!(nt, ow.om.tel.lm.s, ow.om.star.lm.s)
else
    _custom_copy!(nt, ow.om.tel.lm, ow.om.star.lm)
end
ow.o.star .= star_model(ow.om)
ow.o.tel .= tel_model(ow.om)

# optimize RVs
options = Optim.Options(;callback=optim_cb_local, g_tol=g_tol*sqrt(length(ow.rv.p0) / length(ow.telstar.p0)), kwargs...)
ow.om.rv.lm.M .= calc_doppler_component_RVSKL(ow.om.star.λ, ow.om.star.lm.μ)
result_rv, ow.om.rv.lm.s[:] = _OSW_optimize!(ow.rv, options)
ow.o.rv .= rv_model(ow.om, d)
recalc_total!(ow.o, d)


## Downsizing model

if !model.metadata[:todo][:downsized]
    test_n_comp_tel = 0:8
    test_n_comp_star = 0:8
    ks = zeros(Int, length(test_n_comp_tel), length(test_n_comp_star))
    comp_ls = zeros(length(test_n_comp_tel), length(test_n_comp_star))
    for (i, n_tel) in enumerate(test_n_comp_tel)
        for (j, n_star) in enumerate(test_n_comp_star)
            comp_ls[i, j], ks[i, j] = SSOF.test_ℓ_for_n_comps([n_tel, n_star], model, data)
        end
    end
    n_comps_best, ℓ, aic, bic = SSOF.choose_n_comps(comp_ls, ks, test_n_comp_tel, test_n_comp_star, data.var; return_inters=true)
    @save save_path*"model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star

    model_large = copy(model)
    model = SSOF.downsize(model, n_comps_best[1], n_comps_best[2])
    model.metadata[:todo][:downsized] = true
    model.metadata[:todo][:reg_improved] = true
    workspace, loss = SSOF.OptimWorkspace(model, data; return_loss_f=true)
    SSOF.fine_train_OrderModel!(workspace; print_stuff=true)  # 16s
    model.metadata[:todo][:optimized] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel # model_large
end


## Getting RV error bars (only regularization held constant)

if !model.metadata[:todo][:err_estimated]
    data.var[data.var.==Inf] .= 0
    data_noise = sqrt.(data.var)
    data.var[data.var.==0] .= Inf

    data_holder = copy(data)
    model_holder = copy(model)
    n = 50
    rv_holder = zeros(n, length(model.rv.lm.s))
    @time for i in 1:n
        data_holder.flux .= data.flux .+ (data_noise .* randn(size(data_holder.var)))
        SSOF.train_OrderModel!(SSOF.OptimWorkspace(model_holder, data_holder), f_tol=1e-8)
        rv_holder[i, :] = SSOF.rvs(model_holder)
    end
    rv_errors = vec(std(rv_holder; dims=1))
    model.metadata[:todo][:err_estimated] = true
    @save save_path*"results.jld2" model rvs_naive rvs_notel rv_errors
end

## Plots

if save_plots

    include(SSOF_path * "/src/_plot_functions.jl")

    using CSV, DataFrames
    expres_output = CSV.read(SSOF_path * "/EXPRES/" * star * "_activity.csv", DataFrame)
    eo_rv = expres_output."CBC RV [m/s]"
    eo_rv_σ = expres_output."CBC RV Err. [m/s]"
    eo_time = expres_output."Time [MJD]"

    # Compare RV differences to actual RVs from activity
    rvs_notel_opt = SSOF.rvs(model)
    plt = plot_model_rvs_new(times_nu, rvs_notel_opt, vec(rv_errors), eo_time, eo_rv, eo_rv_σ; display_plt=interactive, markerstrokewidth=1);
    png(plt, save_path * "model_rvs.png")

    if !(typeof(model.star.lm) <: SSOF.TemplateModel)
        plt = plot_stellar_model_bases(model; display_plt=interactive);
        png(plt, save_path * "model_star_basis.png")

        plt = plot_stellar_model_scores(model; display_plt=interactive);
        png(plt, save_path * "model_star_weights.png")
    end

    if !(typeof(model.tel.lm) <: SSOF.TemplateModel)
        plt = plot_telluric_model_bases(model; display_plt=interactive);
        png(plt, save_path * "model_tel_basis.png")

        plt = plot_telluric_model_scores(model; display_plt=interactive);
        png(plt, save_path * "model_tel_weights.png")
    end

    plt = status_plot(workspace.o, workspace.d; display_plt=interactive);
    png(plt, save_path * "status_plot.png")

    plt = component_test_plot(ℓ, test_n_comp_tel, test_n_comp_star);
    png(plt, save_path * "l_plot.png")

    plt = component_test_plot(aic, test_n_comp_tel, test_n_comp_star; ylabel="AIC");
    png(plt, save_path * "aic_plot.png")

    plt = component_test_plot(bic, test_n_comp_tel, test_n_comp_star; ylabel="BIC");
    png(plt, save_path * "bic_plot.png")
end

model
Pkg.add("ParameterHandling")
using ParameterHandling

x = ones(5000,10)
s = ones(10,100)
test = SSOF.BaseLinearModel(x, s)
# x[1] = 2
# test.M
using ParameterHandling
@time v, unflatten, unflatten! = flatten(test)
@time v2, unflatten2 = flatten(x)


@time unflatten!(v, test)
@time x[:] = unflatten2(v2)

# comment
workspace.o
using BenchmarkTools
@btime SSOF.loss(workspace.o, workspace.om, workspace.d)
x = SSOF.tel_model(workspace.om)
@btime SSOF.loss(workspace.o, workspace.om, workspace.d; tel=workspace.om.tel.lm)
function loss2(o::SSOF.Output, om::SSOF.OrderModel, d::SSOF.Data;
    recalc_tel::Bool=true, recalc_star::Bool=true, recalc_rv::Bool=true)

    recalc_tel ? tel_o = SSOF.tel_model(om) : tel_o = o.tel
    recalc_star ? star_o = SSOF.star_model(om) : star_o = o.star
    recalc_rv ? rv_o = SSOF.rv_model(om) : rv_o = o.rv
    return SSOF._loss(tel_o, star_o, rv_o, d)
end

@btime loss2(workspace.o, workspace.om, workspace.d)
SSOF.rv_model(workspace.om) == workspace.o.rv
@btime loss2(workspace.o, workspace.om, workspace.d; recalc_tel=false, recalc_star=false, recalc_rv=false)

loss2(workspace.o, workspace.om, workspace.d)
loss2(workspace.o, workspace.om, workspace.d; recalc_tel=false, recalc_star=false, recalc_rv=false)

SSOF._loss(SSOF.tel_model(workspace.om, SSOF.star_model(workspace.om), SSOF.rv_model(workspace.om), workspace.d)

workspace.d
data2 = SSOF.GenericData(data.flux, data.var, data.log_λ_obs, data.log_λ_star)

x = ones(1000, 3)
y = ones(3, 100)
z = zeros(3, 100)
@time lm1 = SSOF.LinearModel(x, y)
lm2 = SSOF.LinearModel(x, z)

lm1.M[1] = 10
lm2.M[1]

_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::GenericData) =
    sum(((total_model(tel, star, rv) .- d.flux) .^ 2) ./ d.var)
# χ² loss function broadened by an lsf at each time
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix, d::LSFData) =
    mapreduce(i -> sum((((d.lsf_broadener[i] * (view(tel, :, i) .* (view(star, :, i) .+ view(rv, :, i)))) .- view(d.flux, :, i)) .^ 2) ./ view(d.var, :, i)), +, 1:size(tel, 2))

@time hcat([rand(1000) for i in 1:200]...)

using SparseArrays
using BandedMatrices
xb = BandedMatrix(zeros(10000, 10000), (13,13))
x = Matrix(xb)
y = ones(10000,1)
@time x * y
@time xs * y
@time xb * y


Pkg.add("JLD2")
using JLD2

@save "test1.jld2" x
@save "test2.jld2" xb
@save "test3.jld2" xs

xs[1] = 1
xs
dropzeros!(xs)

xs

typeof(xs) <: SparseMatrixCSC



SSOF.trapz_small([1, 4], [3, 12])
(4-1) * (3+12) / 2
