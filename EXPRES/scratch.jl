## moved from end of init.jl


# using Plots
#
# snr = sqrt.(tf_data.flux.^2 ./ tf_data.var)
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

# abs_M = abs.(tf_model.tel.lm.M[:, 1])
# plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.975,1))
function proposed_new_cuts(M::AbstractVector, mask_inds::UnitRange, cutoff::Real)
    abs_M = abs.(M)
    # plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.9,1))
    # cdf = [sum(view(abs_M, 1:i)) for i in 1:length(abs_M)] ./ sum(abs_M)
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
    possible_new_inds_tfm = [proposed_new_cuts(M[:, i], mask_inds, cutoff) for i in 1:size(M, 2)]
    possible_new_inds_mask = [proposed_new_inds(possible_new_inds_tfm[i], M[:, i], mask_inds) for i in 1:size(M, 2)]
    return [maximum([inds[1] for inds in possible_new_inds_mask]), minimum([inds[2] for inds in possible_new_inds_mask])],
        [maximum([inds[1] for inds in possible_new_inds_tfm]), minimum([inds[2] for inds in possible_new_inds_tfm])]
end
function new_inds(tfom::SSOF.OrderModel, mask_inds::UnitRange)
    mask_inds_star, _ = new_inds_M(tfom.star.lm.M, mask_inds, 0.975)
    mask_inds_tel, _ = new_inds_M(tfom.tel.lm.M, mask_inds, 0.9975)
    return max(mask_inds_star[1], mask_inds_tel[1]):min(mask_inds_star[2], mask_inds_tel[2])
end
new_inds(tf_model, mask_inds)

proposed_new_cuts(tf_model.star.lm.M[:, 1], mask_inds, 0.975)


abs_M = abs.(tf_model.star.lm.M[:, 1])
plot((1:length(abs_M)) ./ length(abs_M), sort(abs_M); xrange=(0.9,1))
cdf = [sum(view(abs_M, 1:i)) for i in 1:length(abs_M)] ./ sum(abs_M)
plot(cdf)

new_inds_M(tf_model.star.lm.M, mask_inds, 0.9975)
new_inds_M(tf_model.tel.lm.M, mask_inds, 0.9975)

mask_inds
mask_inds = new_inds(tf_model, mask_inds)

# _, tfm_inds = new_inds(tf_model.star.lm.M, mask_inds)
# tf_model.star.lm.M[1:tfm_inds[1], :] .= 0
# tf_model.star.lm.M[tfm_inds[2]:end, :] .= 0
# _, tfm_inds = new_inds(tf_model.tel.lm.M, mask_inds)
# tf_model.tel.lm.M[1:tfm_inds[1], :] .= 0
# tf_model.tel.lm.M[tfm_inds[2]:end, :] .= 0

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
fit_continuum_gif(tf_data.log_λ_obs[:, i], tf_data.flux[:, i], tf_data.var[:, i]; order=6)

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
spectra_interp(tfom.tel(), tfom.lih_t2o)
tf_workspace.tfom.
tf_workspace.tfo.tel
SSOF.tel_model(tf_model)

tf_workspace.tfd.flux

_fracvar(tf_workspace.tfd.flux - tf_workspace.tfo.tel, Y, 1 ./ tf_workspace.tfd.var; var_tot=sum(abs2, X .* weights))

# TODO ERES presentation plots

hmm = status_plot(tf_workspace.tfo, tf_data)
png(hmm, "status_plot")
plot_stellar_model_bases(tf_model; inds=1:3)
hmm = plot_telluric_model_bases(tf_model; inds=1:3)
png(hmm, "telluric_plot")
anim = @animate for i in 1:40
    plt = plot_spectrum(; title="Telluric Spectrum")
    plot!(plt, exp.(tf_data.log_λ_obs[:, i]), view(tf_workspace.tfo.tel, :, i), label="", yaxis=[0.95, 1.005])
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
rvs_notel_opt = (tf_model.rv.lm.s .* light_speed_nu)'
plt = plot_model_rvs_new(times_nu, rvs_notel_opt, rv_errors, eo_time, eo_rv, eo_rv_σ; markerstrokewidth=1, xlim=(58764.35, 58764.40))
scatter(x, x; yerror=x)

scatter(eo_time, eo_time-times_nu)
histogram(eo_time - times_nu)
