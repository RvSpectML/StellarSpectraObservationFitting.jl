# using Pkg
# Pkg.activate("EXPRES")
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Finding LSF width as a function of λ
using Distributions
using LinearAlgebra
_recalc = true
_inter_poly_order = 2
_intra_poly_order = 2

if _recalc
    use_wavenumber = true; use_log = true
    using CSV, DataFrames
    eo = CSV.read("D:/Christian/Downloads/expres_psf.txt", DataFrame)
    # eo = CSV.read("C:/Users/chris/Downloads/expres_psf.txt", DataFrame)
    filter!(:line => ==("LFC"), eo)
    sort!(eo, ["wavenumber [1/cm]"])

    if use_wavenumber
        λs_func(wn) = wn[:]
        dλs_func(wn) = ones(length(wn))
        unit_str = "1/cm"
        xlab = "Wavenumber ($unit_str)"
    else
        use_log ? λs_func(wn) = log.(1e8 ./ wn) : λs_func(wn) = 1e8 ./ wn
        use_log ? dλs_func(wn) = -1 ./ wn : dλs_func(wn) = -1e8 ./ (wn .^ 2)
        unit_str = "Å?"
        xlab = "Wavelength ($unit_str)"
    end
    λs = λs_func(eo."wavenumber [1/cm]")
    σ = fwhm_2_σ(λs .* (eo."fwhm [1/cm]" ./ eo."wavenumber [1/cm]"))

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
    _w = SSOF.general_lst_sq(dm, σ, (dλs_func(eo."wavenumber [1/cm]") .* eo."fwhm error [1/cm]") .^ 2)  # note the errors are not scaled FWHM -> σ
    println("new lsf_σ w: ", _w)
    model = dm * _w
else
    _w = [-0.00047152836831767077, 0.00011337984500164679, -6.412301552294912e-6, -0.00020175981792570192, 0.015969041179354226]
end

lsf_σ(logλ::Real, logλ_m_order_mean::Real) = [1, logλ, logλ*logλ, logλ_m_order_mean, logλ_m_order_mean*logλ_m_order_mean]' * _w
function lsf_σ(logλ::AbstractVector, logλ_m_order_mean::AbstractVector)
    dm = ones(length(logλ), 3 + _poly_order)
    dm[:, 2] = logλ
    dm[:, 3] = logλ .* logλ
    dm[:, 4] = logλ_m_order_mean
    for i in 2:_poly_order
        dm[:, i+3] = logλ_m_order_mean .^ i
    end
    return dm * _w
end

# plt = my_scatter(λs, σ; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# png(plt, "expres_lsf")
#
# inds1 = eo.order .== 38
# inds2 = eo.order .== 39
# plt = my_scatter(λs[inds1], σ[inds1]; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(λs[inds2], σ[inds2]; label="")
# png(plt, "expres_lsf_zoom")
#
plt = my_scatter(λs, σ; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
for order in orders
    inds_temp = eo.order .== order
    plot!(λs[inds_temp], model[inds_temp]; label="", lw=4)
end
plt
png(plt, "expres_lsf_model")
#
# plt = my_scatter(λs[inds1], σ[inds1]; label="", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(λs[inds2], σ[inds2]; label="")
# plot!(λs[inds1], model[inds1]; label="model", lw=6, c=plt_colors[6])
# plot!(λs[inds2], model[inds2]; label="model", lw=6)
# png(plt, "expres_lsf_model_zoom")
#
resids = σ ./ model
std(resids)
# plt = my_scatter(λs, resids; label="data / model", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# png(plt, "expres_lsf_resids")
# plt = scatter(λs[inds1], resids[inds1]; label="data / model", xlabel=xlab, ylabel="LSF σ ($unit_str)")
# scatter!(λs[inds2], resids[inds2]; label="data / model")
# png(plt, "expres_lsf_resids_zoom")
