## moved from end of init.jl


# using Plots
#
# snr = sqrt.(data.flux.^2 ./ data.var)
# heatmap(snr)
# histogram(collect(Iterators.flatten(snr)))
#
# plt = plot(;xlabel="MJD", ylabel="Å", title="EXPRES wavelength calibration ($star, order $desired_order)")
# for i in 1:10
#     scatter!(plt, times_nu, exp.(log_λ_obs[i, :]); label="pixel $(mask_inds[i])")
# end
# display(plt)
# png("obs.png")
#
# plt = plot(;xlabel="MJD", ylabel="Å", title="EXPRES wavelength calibration ($star, order $desired_order)")
# for i in 1:10
#     scatter!(plt, times_nu, exp.(log_λ_star[i, :]); label="pixel $(mask_inds[i])")
# end
# display(plt)
# png("bary.png")

# inst = all_spectra[1].inst
# extra_chop = 0
# mask_inds = (min_col_default(inst, desired_order) + extra_chop):(max_col_default(inst, desired_order) - extra_chop)  # 850:6570
# mask_inds = 2270:5150

# using Plots
# plot(all_spectra[1].metadata[:tellurics][:, desired_order])

# abs_M = abs.(model.tel.lm.M[:, 1])
# plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.975,1))
function proposed_new_cuts(M::AbstractVector, mask_inds::UnitRange, cutoff::Real)
    abs_M = abs.(M)
    # plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.9,1))
    # cdf = [sum(view(abs_M, 1:i)) for i in eachindex(abs_M)] ./ sum(abs_M)
    # plot(cdf)
    @assert 0 < cutoff < 1
    high_M = quantile(abs_M, cutoff)
    half = Int(floor(length(abs_M) / 2))
    cuts = [1, length(abs_M)]
    cut_low = findlast((x) -> x > high_M, abs_M[1:half])
    cut_high = findfirst((x) -> x > high_M, abs_M[(half+1):end])
    if cut_low!=nothing; cuts[1] = cut_low + 1 end
    if cut_high!=nothing; cuts[2] = half + cut_high - 1 end
    return cuts
end
function proposed_new_inds(cuts::Vector, M::AbstractVector, mask_inds::UnitRange)
    # println(maximum(abs_M) / high_M)
    cuts2 = Int.(round.(length(mask_inds) .* (cuts ./ length(M))))
    return [mask_inds[1] + cuts2[1], mask_inds[1] + cuts2[2]]
end
function new_inds_M(M::AbstractArray, mask_inds::UnitRange, cutoff::Real)
    possible_new_inds_m = [proposed_new_cuts(M[:, i], mask_inds, cutoff) for i in 1:size(M, 2)]
    possible_new_inds_mask = [proposed_new_inds(possible_new_inds_m[i], M[:, i], mask_inds) for i in 1:size(M, 2)]
    return [maximum([inds[1] for inds in possible_new_inds_mask]), minimum([inds[2] for inds in possible_new_inds_mask])],
        [maximum([inds[1] for inds in possible_new_inds_m]), minimum([inds[2] for inds in possible_new_inds_m])]
end
function new_inds(om::SSOF.OrderModel, mask_inds::UnitRange)
    mask_inds_star, _ = new_inds_M(om.star.lm.M, mask_inds, 0.975)
    mask_inds_tel, _ = new_inds_M(om.tel.lm.M, mask_inds, 0.9975)
    return max(mask_inds_star[1], mask_inds_tel[1]):min(mask_inds_star[2], mask_inds_tel[2])
end
new_inds(model, mask_inds)

proposed_new_cuts(model.star.lm.M[:, 1], mask_inds, 0.975)


abs_M = abs.(model.star.lm.M[:, 1])
plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.9,1))
cdf = [sum(view(abs_M, 1:i)) for i in eachindex(abs_M)] ./ sum(abs_M)
plot(cdf)

new_inds_M(model.star.lm.M, mask_inds, 0.9975)
new_inds_M(model.tel.lm.M, mask_inds, 0.9975)

mask_inds
mask_inds = new_inds(model, mask_inds)

# _, m_inds = new_inds(model.star.lm.M, mask_inds)
# model.star.lm.M[1:m_inds[1], :] .= 0
# model.star.lm.M[m_inds[2]:end, :] .= 0
# _, m_inds = new_inds(model.tel.lm.M, mask_inds)
# model.tel.lm.M[1:m_inds[1], :] .= 0
# model.tel.lm.M[m_inds[2]:end, :] .= 0

# using BenchmarkTools
# using LinearAlgebra
# hmm = ones(1000)
# x = diagm(hmm)
# xc = cholesky(x)
# @btime cholesky(x)
# @btime x \ hmm
# @btime xc \ hmm
# y = Diagonal(hmm)
# yc = cholesky(y)
# @btime cholesky(y)
# @btime y \ hmm
# @btime yc \ hmm
# using BandedMatrices
# zz = Dict(0=> ones(1000), -1=> zeros(999), 1=> zeros(999))
# z = BandedMatrix(z)
# @btime cholesky(z)
# @btime z \ hmm

##  Continuum GIF, do before SSOF.process!
function fit_continuum_gif(x::AbstractVector, y::AbstractVector, σ²::AbstractVector; order::Int=6, nsigma::Vector{<:Real}=[0.3,3.0], maxniter::Int=50, filename::String="show_continuum_fit.gif")
    A = StellarSpectraObservationFitting.vander(x .- mean(x), order)
    m = fill(true, length(x))
    μ = ones(length(x))
	ex = exp.(x)
    anim = @animate for i in 1:maxniter
        m[σ² .== Inf] .= false  # mask out the bad pixels
        w = StellarSpectraObservationFitting.general_lst_sq(view(A, m, :), view(y, m), view(σ², m))
        μ[:] = A * w
		plt = plot_spectrum(; ylabel = "Blaze Normalized Flux")
		my_scatter!(plt, ex[m], y[m]; label="")
		my_scatter!(plt, ex[.!m], y[.!m]; label="")
		plot!(plt, ex, μ; label="")
        resid = y - μ
        # sigma = median(abs.(resid))
		sigma = std(resid)
        m_new = (-nsigma[1]*sigma) .< resid .< (nsigma[2]*sigma)
        if sum(m) == sum(m_new); break end
        m = m_new
    end
	gif(anim, filename, fps = 3)
    return μ
end

include("../src/_plot_functions.jl")
i=10
fit_continuum_gif(data.log_λ_obs[:, i], data.flux[:, i], data.var[:, i]; order=6)

## Wavelength calibration plot
plt = _my_plot(;xlabel="MJD", ylabel="Observer Frame Wavelength (Å)", title="Wavelength Calibration", thickness_scaling=3)
for i in 1:5
    scatter!(plt, times_nu, exp.(log_λ_obs[i, :]); label="Pixel $(mask_inds[i])", markerstrokewidth=0)
end
display(plt)
png("obs.png")

plt = _my_plot(;xlabel="MJD", ylabel="Barycentric Frame Wavelength (Å)", title="Wavelength Calibration", thickness_scaling=3, markerstrokestyle=:dash)
for i in 1:5
    scatter!(plt, times_nu, exp.(log_λ_star[i, :]); label="Pixel $(mask_inds[i])", markerstrokewidth=0)
end
display(plt)
png("bary.png")

## moved from end of analysis.jl

# TODO on choosing amount of basis vectors
spectra_interp(om.tel(), om.lih_t2o)
workspace.om.
workspace.o.tel
SSOF.tel_model(model)

workspace.d.flux

_fracvar(workspace.d.flux - workspace.o.tel, Y, 1 ./ workspace.d.var; var_tot=sum(abs2, X .* weights))

# TODO ERES presentation plots

hmm = status_plot(workspace)
png(hmm, "status_plot")
plot_stellar_model_bases(model; inds=1:3)
hmm = plot_telluric_model_bases(model; inds=1:3)
png(hmm, "telluric_plot")
anim = @animate for i in 1:40
    plt = plot_spectrum(; title="Telluric Spectrum")
    plot!(plt, exp.(data.log_λ_obs[:, i]), view(workspace.o.tel, :, i), label="", yaxis=[0.95, 1.005])
end
gif(anim, "show_telluric_var.gif", fps = 10)

# why are times different?
using CSV, DataFrames
expres_output = CSV.read(expres_data_path * star * "_activity.csv", DataFrame)
eo_rv = expres_output."CBC RV [m/s]"
eo_rv_σ = expres_output."CBC RV Err. [m/s]"
eo_time = expres_output."Time [MJD]"

# Compare RV differences to actual RVs from activity
include("../src/_plot_functions.jl")
rvs_notel_opt = SSOF.rvs(model)
plt = plot_model_rvs(times_nu, rvs_notel_opt, rv_errors, eo_time, eo_rv, eo_rv_σ; markerstrokewidth=1, xlim=(58764.35, 58764.40))
scatter(x, x; yerror=x)

scatter(eo_time, eo_time-times_nu)
histogram(eo_time - times_nu)


## elbow testing

using Polynomials

function elbow(x::AbstractVector, y::AbstractVector)
    poly = Polynomials.fit(x, y, length(x)-1)
    d2poly = derivative(poly, 2)
    d3poly = derivative(d2poly)
    tester = append!(float.([x[1], x[end]]), roots(d3poly))
    return tester[argmax(abs.(d2poly.(tester)))]
end

Int.([elbow(test_n_comp_star, comp_ℓs[i, :]) for i in eachindex(test_n_comp_tel)])  # number of suggested stellar components
Int.([elbow(test_n_comp_tel, comp_ℓs[:, i]) for i in eachindex(test_n_comp_star)])  # number of suggested telluric components

pltx = test_n_comp_tel[1]-1:0.1:test_n_comp_tel[end]+1
poly = Polynomials.fit(test_n_comp_tel, comp_ℓs[:, 1], 5)
scatter(test_n_comp_tel, comp_ℓs[:, 1])
plot!(pltx, poly.(collect(pltx)))
# plot!(pltx, derivative(poly, 1).(collect(pltx)))
plot!(pltx, derivative(poly, 2).(collect(pltx)))
plot!(pltx, derivative(poly, 3).(collect(pltx)))
# plot!(pltx, derivative(poly, 4).(collect(pltx)))


## Fixing Manifest.jl

using Pkg
Pkg.activate(".")
Pkg.precompile()
Pkg.update()

Pkg.activate("EXPRES")
Pkg.instantiate()

Pkg.rm("EMPCA")
Pkg.rm("StellarSpectraObservationFitting")
Pkg.rm("RvSpectMLBase")
Pkg.rm("RvSpectML")
Pkg.rm("EchelleCCFs")
Pkg.rm("EchelleInstruments")

Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\EMPCA.jl")
Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\StellarSpectraObservationFitting")
# Pkg.develop(;path="D:\\Christian\\Documents\\GitHub\\EMPCA")
# Pkg.develop(;path="C:\\Users\\Christian\\Dropbox\\GP_research\\julia\\StellarSpectraObservationFitting")
# Pkg.develop(;path="/storage/work/c/cjg66/GP_research/EMPCA/")
# Pkg.develop(;path="/storage/work/c/cjg66/GP_research/StellarSpectraObservationFittin.jl/")
Pkg.develop(;url="https://github.com/christiangil/RvSpectMLBase.jl")
Pkg.develop(;url="https://github.com/christiangil/RvSpectML.jl")
Pkg.develop(;url="https://github.com/christiangil/EchelleCCFs.jl")
Pkg.develop(;url="https://github.com/christiangil/EchelleInstruments.jl")

Pkg.instantiate()
import Pkg; Pkg.precompile()

using CSV

Pkg.rm("GLOM_RV_Example")
Pkg.rm("GPLinearODEMaker")
Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\GPLinearODEMaker.jl")
Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\GLOM_RV_Example")


## using sparse arrays is 2x faster (but takes twice the memory) than banded for multiplication
using SparseArrays
using BandedMatrices
xb = BandedMatrix(ones(10000, 10000), (13,13))
xs = sparse(xb)
x = Matrix(xb)
y = ones(10000,1)
@time x * y
@time xs * y
@time xb * y

@save "test1.jld2" x
@save "test2.jld2" xb
@save "test3.jld2" xs

## new spectra interp is faster

using SparseArrays
function spectra_interp_old(model::AbstractMatrix, interp_helper::Vector{SparseMatrixCSC})
	interped_model = zeros(size(interp_helper[1], 1), size(model, 2))
	@inbounds @simd for i in 1:size(model, 2)
		interped_model[:, i] .= interp_helper[i] * view(model, :, i)
	end
	return interped_model
end
spectra_interp(model::AbstractMatrix, interp_helper::Vector{SparseMatrixCSC}) =
	hcat([interp_helper[i] * view(model, :, i) for i in 1:size(model, 2)]...)

x = ts.θ.tel()

@btime spectra_interp(x, workspace.om.t2o)
@btime spectra_interp_old(x, workspace.om.t2o)


broadener = randn(100,100)
model = ones(100,10)
model2 = copy(model)
model3 = copy(model)

@time for i in 1:size(model, 2)
	model[:,i] .= broadener * view(model, :,i)
end
@time model .= broadener * model
all(isapprox.(model2,model3))
model2 .= broadener * model

@time broadener * model


# inds = 5:24
# heatmap(exp.(data.log_λ_obs[inds,1]),exp.(data.log_λ_obs[inds,1]),data.lsf[1][inds,inds])
# png("lsf1")
# inds = size(data.log_λ_obs,1)-23:size(data.log_λ_obs,1)-4
# heatmap(exp.(data.log_λ_obs[inds,1]),exp.(data.log_λ_obs[inds,1]),data.lsf[1][inds,inds])
# png("lsf2")

## Looking at interpolation matrices

heatmap(Array(model.b2o[1][1:1000, 1:1000]))

hmm = hcat([Array(model.b2o[i][100,:]) for i in 1:114]...)
shift = round(Int, 100*2*sqrt(2))
model_inds = 50+shift:370+shift
plt = heatmap(1:114, model_inds, hmm[model_inds, :]; xlabel="Time", ylabel="Model Pixels", title="Star Model ➡ Obs Pixel Interpolation Map (Pix 100)")
png(plt, "b2o_heatmap")
plt = scatter(exp.(data.log_λ_star[99:101,:]'); label="Pix 99-101", xlabel="Time", ylabel="Obs Bary λ (Å)")
png(plt, "b2o_scatter")

hmm = hcat([Array(model.t2o[i][100,:]) for i in 1:114]...)
model_inds = 50+shift:100+shift
plt = heatmap(1:114, model_inds, hmm[model_inds, :]; xlabel="Time", ylabel="Model Pixels", title="Tel Model ➡ Obs Pixel Interpolation Map (Pix 100)")
png(plt, "t2o_heatmap")
plt = scatter(exp.(data.log_λ_obs[99:101,:]'); label="Pix 99-101", xlabel="Time", ylabel="Obs λ (Å)")
png(plt, "t2o_scatter")
