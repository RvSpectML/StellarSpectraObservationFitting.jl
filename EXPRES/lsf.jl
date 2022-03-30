# using Pkg
# Pkg.activate("EXPRES")
import StellarSpectraObservationFitting as SSOF

## Finding LSF width as a function of wavenumber
using Distributions
using SparseArrays
using LinearAlgebra
_recalc = false
_inter_poly_order = 2
_intra_poly_order = 2

if _recalc
    using CSV, DataFrames
    eo = CSV.read("D:/Christian/Downloads/expres_psf.txt", DataFrame)
    # eo = CSV.read("C:/Users/chris/Downloads/expres_psf.txt", DataFrame)
    filter!(:line => ==("LFC"), eo)
    sort!(eo, ["wavenumber [1/cm]"])

    unit_str = "1/cm"
    xlab = "Wavenumber ($unit_str)"
    wns = eo."wavenumber [1/cm]"
    σ = SSOF.fwhm_2_σ.(eo."fwhm [1/cm]")

    dm = ones(size(eo, 1), 1 + _inter_poly_order + _intra_poly_order)
    # orders = [i for i in 1:100 if i in eo.order]
    orders = minimum(eo.order):maximum(eo.order)
    @assert _inter_poly_order > 0
    @assert _intra_poly_order > 0
    for order in orders
        inds_temp = eo.order .== order
        df = filter(:order => ==(order), eo)
        dm[inds_temp, 2] = λs_func(df."wavenumber [1/cm]")
        for i in 2:_inter_poly_order
            dm[inds_temp, i + 1] = dm[inds_temp, 2] .^ i
        end
        dm[inds_temp, _inter_poly_order + 2] = dm[inds_temp, 2] .- mean(dm[inds_temp, 2])
        for i in 2:_intra_poly_order
            dm[inds_temp, i + 1 + _inter_poly_order] = dm[inds_temp, _inter_poly_order + 2] .^ i
        end
    end
    _w = SSOF.general_lst_sq(dm, σ, (eo."fwhm error [1/cm]") .^ 2)  # note the errors are not scaled FWHM -> σ
    println("new lsf_σ w: ", _w)
    model = dm * _w
    _min_wn, _max_wn = extrema(wns)
else
    _w = [-0.013170986773323784, 4.8136021892119126e-6, -3.772787891582741e-11, 2.3266186490906004e-5, 1.100967064582426e-7]
    _min_wn, _max_wn = 13760.52558749721, 20111.11872452446
end

lsf_σ_inter_order(wn::Real) = _w[1] + wn*(_w[2] + _w[3]*wn)
lsf_σ_intra_order(wn_m_order_mean::Real) = wn_m_order_mean*(_w[4] + _w[5]*wn_m_order_mean)
lsf_σ(wn::Real, wn_m_order_mean::Real) = lsf_σ_inter_order(wn) + lsf_σ_intra_order(wn_m_order_mean)
function lsf_σ_safe(wn::Real, wn_m_order_mean::Real)
    if _min_wn < wn < _max_wn
        return lsf_σ(wn, wn_m_order_mean)
    else
        return lsf_σ_inter_order(wn)
    end
end
function lsf_σ_inter_order(wn::AbstractVector)
    dm = ones(length(wn), 3)
    dm[:, 2] = wn
    dm[:, 3] = wn .* wn
    return dm * view(_w, 1:3)
end

function lsf_σ(wn::AbstractVector, wn_m_order_mean::AbstractVector)
    dm = ones(length(wn), 1 + _inter_poly_order + _intra_poly_order)
    dm[:, 2] = wn
    for i in 2:_inter_poly_order
        dm[:, i+1] = wn .^ i
    end
    dm[:, 2+_inter_poly_order] = wn_m_order_mean
    for i in 2:_intra_poly_order
        dm[:, i+_inter_poly_order+1] = wn_m_order_mean .^ i
    end
    return dm * _w
end
function lsf_σ_safe(wn::AbstractVector, wn_m_order_mean::AbstractVector)
    if _min_wn < mean(wn) < _max_wn
        return lsf_σ(wn, wn_m_order_mean)
    else
        return lsf_σ_inter_order(wn)
    end
end

# plt = _scatter(wns, σ; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# png(plt, "expres_lsf")
#
# inds1 = eo.order .== 38
# inds2 = eo.order .== 39
# plt = _scatter(wns[inds1], σ[inds1]; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(wns[inds2], σ[inds2]; label="")
# png(plt, "expres_lsf_zoom")
#
# plt = _scatter(wns, σ; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# for order in orders
#     inds_temp = eo.order .== order
#     plot!(wns[inds_temp], model[inds_temp]; label="", lw=4)
# end
# plt
# png(plt, "expres_lsf_model")
#
# plt = _scatter(wns[inds1], σ[inds1]; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(wns[inds2], σ[inds2]; label="")
# plot!(wns[inds1], model[inds1]; label="model", lw=6, c=plt_colors[6])
# plot!(wns[inds2], model[inds2]; label="model", lw=6)
# png(plt, "expres_lsf_model_zoom")
#
# resids = σ ./ model
# std(resids)
# plt = _scatter(wns, resids; label="data / model", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# png(plt, "expres_lsf_resids")
# plt = scatter(wns[inds1], resids[inds1]; label="data / model", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(wns[inds2], resids[inds2]; label="data / model")
# png(plt, "expres_lsf_resids_zoom")

function EXPRES_lsf(λ::AbstractVector; safe::Bool=true)
    if safe; @assert 1000 < mean(λ) < 50000  "Are you sure you're using λ (Å) and not wavenumber (1/cm) or log(λ)?" end
    wn = SSOF.Å_to_wavenumber.(λ)
    σs = lsf_σ_safe(wn, wn .- mean(wn))
    nwn = -wn
    holder = zeros(length(nwn), length(nwn))
    # max_w = 0
    for i in eachindex(nwn)
        lo, hi = SSOF.searchsortednearest(nwn, [nwn[i] - 3 * σs[i], nwn[i] + 3 * σs[i]])
        lsf = Normal(wn[i], σs[i])
        holder[i, lo:hi] = pdf.(lsf, wn[lo:hi])
        holder[i, lo:hi] ./= sum(view(holder, i, lo:hi))
        # max_w = max(max_w, max(hi-i, i-lo))
    end
    ans = sparse(holder)
    dropzeros!(ans)
    return ans
    # return BandedMatrix(holder, (max_w, max_w))
end
# EXPRES_lsfs(λ::AbstractMatrix; kwargs...) =
#     [lsf(view(λ, :, i); kwargs...) for i in 1:size(λ, 2)]
EXPRES_lsf(λ::AbstractMatrix; kwargs...) =
    EXPRES_lsf(vec(median(λ; dims=2)); kwargs...)
