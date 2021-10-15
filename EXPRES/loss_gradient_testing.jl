## Importing packages
using Pkg
Pkg.activate("EXPRES")

using JLD2
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
import StatsBase


Pkg.resolve()
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

inds = 5:24
heatmap(exp.(data.log_λ_obs[inds,1]),exp.(data.log_λ_obs[inds,1]),data.lsf_broadener[1][inds,inds])
png("lsf1")
inds = size(data.log_λ_obs,1)-23:size(data.log_λ_obs,1)-4
heatmap(exp.(data.log_λ_obs[inds,1]),exp.(data.log_λ_obs[inds,1]),data.lsf_broadener[1][inds,inds])
png("lsf2")


# 7020, 114
ind_λ = 1:7020; ind_t = 1:114
data_small = SSOF.LSFData(data.flux[ind_λ, ind_t], data.var[ind_λ, ind_t], data.log_λ_obs[ind_λ, ind_t], data.log_λ_star[ind_λ, ind_t], [data.lsf_broadener[i][ind_λ,ind_λ] for i in ind_t])

if false#isfile(save_path*"results.jld2")
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
    @time model = SSOF.OrderModel(data_small, "EXPRES", desired_order, star; n_comp_tel=3, n_comp_star=3, upscale=model_upscale)
    @time rvs_notel, rvs_naive, _, _ = SSOF.initialize!(model, data_small; use_gp=true)
    if !use_reg
        SSOF.zero_regularization(model)
        model.metadata[:todo][:reg_improved] = true
    end
    # @save save_path*"results.jld2" model rvs_naive rvs_notel
end


## Creating optimization workspace
workspace, loss = SSOF.OptimWorkspace(model, data_small; return_loss_f=true)

workspace.telstar.p0

## loss testing

using Zygote
using ParameterHandling
using BenchmarkTools
ts = workspace.telstar
# @time ts.obj.f(ts.p0)
# @time Zygote.gradient(ts.obj.f, ts.p0)

nλ = size(data_small.flux, 1)

# No LSF and χ²
# _loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix) =
#     sum(((SSOF.total_model(tel, star, rv) .- workspace.d.flux) .^ 2) ./ workspace.d.var)
# LSF and χ²
_loss(tel::AbstractMatrix, star::AbstractMatrix, rv::AbstractMatrix) =
    mapreduce(i -> sum((((workspace.d.lsf_broadener[i] * (view(tel, :, i) .* (view(star, :, i) .+ view(rv, :, i)))) .- view(workspace.d.flux, :, i)) .^ 2) ./ view(workspace.d.var, :, i)), +, 1:size(tel, 2))

# Interpolation
# loss2(om::SSOF.OrderModel; tel::SSOF.LinearModel=om.tel.lm, star::SSOF.LinearModel=om.star.lm) =
# 	_loss(SSOF.tel_model(om; lm=tel), SSOF.star_model(om; lm=star), workspace.o.rv)
# no Interpolation
loss2(om::SSOF.OrderModel; tel::SSOF.LinearModel=om.tel.lm, star::SSOF.LinearModel=om.star.lm) =
	_loss(tel()[1:nλ, :], star()[1:nλ, :], workspace.o.rv)

# Adding priors
l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{SSOF.LinearModel, SSOF.LinearModel}}) =
	loss2(workspace.om; tel=nt.tel, star=nt.star) + SSOF.model_prior(nt.tel, workspace.om.reg_tel) + SSOF.model_prior(nt.star, workspace.om.reg_star)
# No priors
# l_telstar(nt::NamedTuple{(:tel, :star,),<:Tuple{SSOF.LinearModel, SSOF.LinearModel}}) =
# 	loss2(workspace.om; tel=nt.tel, star=nt.star)

p0, unflatten = flatten(ts.θ)  # unflatten returns NamedTuple of untransformed params
f = l_telstar ∘ unflatten
f(p0)
@time Zygote.gradient(f, p0)



########## ONLY USELESS GARBAGE BELOW
## one parameter
function g_test(inps, l)
	p0, unflatten = flatten(inps)  # unflatten returns NamedTuple of untransformed params
	f = l ∘ unflatten
	return unflatten(only(Zygote.gradient(f, p0))), f, p0
end

f2(lm) = sum(lm())
g_test(model.tel.lm, f2)

x = copy(model.tel.lm)
x.M .= 0; x.μ .= 0; x.M[1000] = 1;

f2(lm) = sum(lm())
g_test(model.tel.lm, f2).M[1000]
sum(x.M * x.s)

f2(lm) = sum(model.t2o[1] * lm())
g_test(model.tel.lm, f2).M[1000]
sum(model.t2o[1] * (x.M * x.s))

sm = SSOF.star_model(model) + SSOF.rv_model(model)
f2(lm) = sum((model.t2o[1] * lm() .* sm))
g_test(model.tel.lm, f2).M[1000]
sum((model.t2o[1] * (x.M * x.s)) .* sm)

f2(lm) = sum(data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm))
g_test(model.tel.lm, f2).M[1000]
sum(data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm))

f2(lm) = sum(data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux)
g_test(model.tel.lm, f2).M[1000]
sum(data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm))

f2(lm) = sum((data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux) .^ 2)
g_test(model.tel.lm, f2).M[1000]
β = (data_small.lsf_broadener[1] * (model.t2o[1] * model.tel.lm() .* sm) - data_small.flux)
dβ = data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm)
sum(2 .* β .* dβ)

f2(lm) = sum(((data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux) .^ 2) ./ data_small.var)
g_test(model.tel.lm, f2).M[1000]
sum(2 .* β .* dβ ./ data_small.var)

## all M

x = copy(model.tel.lm)
x.μ .= 0; x.M .= 0; x.M[116,3] = 1;

(x.M * x.s)
(x.M * x.s)[116,:]
x.s[3, :]
f2(lm) = sum(lm())
@time g_test(model.tel.lm, f2).M
@time repeat(sum(x.s, dims=2)', size(x.M, 1))


f2(lm) = sum(model.t2o[1] * lm())
@btime g_test(model.tel.lm, f2).M
# sum(model.t2o[1] * (x.M * x.s))
dM = zeros(size(x.M))
mapreduce((i,j) -> i*j, +, 1:size(dM, 1), 1:size(dM, 2))

repeat(1:size(dM, 1), size(dM, 2))

1:size(dM, 2)
1:size(dM, 1)
sum([i^2 for i in 1:8])

hcat([interp_helper[i] * view(model, :, i) for i in 1:size(model, 2)]...)
@btime for i in 1:size(dM, 1)
	for j in 1:size(dM, 2)
		dM[i,j] = sum(kron(view(model.t2o[1], :, i),view(x.s, j, :)))
	end
end
dM

i=100;j=3;
@time sum(view(model.t2o[1], :, i)*transpose(view(x.s, j, :)))
@time sum(view(model.t2o[1], :, i)*view(x.s, j, :)')
@time sum(view(model.t2o[1], :, i).*view(x.s, j, :)')
@time sum(kron(view(model.t2o[1], :, i),view(x.s, j, :)))


sm = SSOF.star_model(model) + SSOF.rv_model(model)
f2(lm) = sum((model.t2o[1] * lm() .* sm))
@time g_test(model.tel.lm, f2).M[1000]
sum((model.t2o[1] * (x.M * x.s)) .* sm)

f2(lm) = sum(data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm))
@time g_test(model.tel.lm, f2).M[1000]
sum(data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm))

f2(lm) = sum(data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux)
@time g_test(model.tel.lm, f2).M[1000]
sum(data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm))

f2(lm) = sum(abs2, data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux)
@time g_test(model.tel.lm, f2).M[1000]
β = (data_small.lsf_broadener[1] * (model.t2o[1] * model.tel.lm() .* sm) - data_small.flux)
dβ = data_small.lsf_broadener[1] * ((model.t2o[1] * (x.M * x.s)) .* sm)
sum(2 .* β .* dβ)

f2(lm) = sum(((data_small.lsf_broadener[1] * (model.t2o[1] * lm() .* sm) - data_small.flux) .^ 2) ./ data_small.var)
@time _, f3, p0 = g_test(model.tel.lm, f2)
sum(2 .* β .* dβ ./ data_small.var)

@time Zygote.gradient(f3, p0)

##
using Plots

y = 3 .+ randn(7)

plot(y; ylims=(0,5))
png("test")
