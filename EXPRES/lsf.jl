# using Pkg
# Pkg.activate("EXPRES")
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Finding LSF width as a function of λ
using Distributions
using LinearAlgebra
_recalc = true
_poly_order = 2

if _recalc
    use_wavenumber = false; use_log = true
    using CSV, DataFrames
    # eo = CSV.read("D:/Christian/Downloads/expres_psf.txt", DataFrame)
    eo = CSV.read("C:/Users/chris/Downloads/expres_psf.txt", DataFrame)
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
    FWHM = λs .* (eo."fwhm [1/cm]" ./ eo."wavenumber [1/cm]")

    dm = ones(size(eo, 1), 3 + _poly_order)
    orders = [i for i in 1:100 if i in eo.order]
    for order in orders
        inds_temp = eo.order .== order
        df = filter(:order => ==(order), eo)
        dm[inds_temp, 2] = λs_func(df."wavenumber [1/cm]")
        dm[inds_temp, 3] = dm[inds_temp, 2] .^ 2
        dm[inds_temp, 4] = dm[inds_temp, 2] .- mean(dm[inds_temp, 2])
        for i in 2:_poly_order
            dm[inds_temp, i+3] = dm[inds_temp, 4] .^ i
        end
    end
    _w = SSOF.general_lst_sq(dm, FWHM, (dλs_func(eo."wavenumber [1/cm]") .* eo."fwhm error [1/cm]") .^ 2)
end
else
    _w = [-0.001110364451108748, 0.00026698913115850097, -1.509981619826013e-5, -0.0004751080635331538, 0.037604218270268254]
end

const fwhm_2_σ_factor = 1 / (2 * sqrt(2 * log(2)))
fwhm_2_σ(fwhm::Real) = fwhm_2_σ_factor * fwhm

lsf_width_log_λ(logλ::Real, logλ_m_order_mean::Real) = fwhm_2_σ[1, logλ, logλ*logλ, logλ_m_order_mean, logλ_m_order_mean*logλ_m_order_mean]' * w
function lsf_width_log_λ(logλ::AbstractVector, logλ_m_order_mean::AbstractVector)
    dm = ones(length(logλ), 3 + _poly_order)
    dm[:, 2] = logλ
    dm[:, 3] = logλ .* logλ
    dm[:, 4] = logλ_m_order_mean
    for i in 2:_poly_order
        dm[:, i+3] = logλ_m_order_mean .^ i
    end
    return fwhm_2_σ.(dm * w)
end
