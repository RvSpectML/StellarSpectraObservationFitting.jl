using Pkg
Pkg.activate("NEID")
import StellarSpectraObservationFitting as SSOF

using FITSIO
using SpecialFunctions
using SparseArrays
function conv_gauss_tophat(x::Real, sigma::Real, boxhalfwidth::Real; amp::Real=1)
    scale = 1 / (sqrt(2) * sigma)
    arg1 = (boxhalfwidth + x) * scale
    arg2 = (boxhalfwidth - x) * scale
    return abs(amp * (erf(arg1) + erf(arg2)))
    # return abs(amp * (erf(arg1) + erf(arg2)) / (2 * erf(boxhalfwidth * scale)))  # used for some visual diagnostic
end
σs = read(FITS("NEID/sigma_arr.fits")[1])
bhws = read(FITS("NEID/boxhalfwidth_arr.fits")[1])
no_lsf_orders = [all(iszero.(view(bhws, :, i))) for i in 1:size(bhws,2)]
@assert all(no_lsf_orders .== [all(iszero.(view(σs, :, i))) for i in 1:size(σs,2)])

three_sigma(σ::Real, bhw::Real) = 3 * abs(σ) + 0.87 * abs(bhw)
function NEID_lsf(order::Int)
    @assert 1 <= order <= length(no_lsf_orders)
    if no_lsf_orders[order]; return nothing end
    n = size(σs, 1)
    holder = zeros(n, n)
    for i in 1:n
        lo = max(1, Int(round(i - three_sigma(σs[i, order], bhws[i, order]))))
        hi = min(n, Int(round(i + three_sigma(σs[i, order], bhws[i, order]))))
        holder[i, lo:hi] = conv_gauss_tophat.((lo-i):(hi-i), σs[i, order], bhws[i, order])
        holder[i, lo:hi] ./= sum(view(holder, i, lo:hi))
    end
    ans = sparse(holder)
    dropzeros!(ans)
    return ans
end

# s = NEID_lsf(100)
# heatmap(Matrix(s[1:100,1:100]))
# heatmap(Matrix(s[end-100:end,end-100:end]))
#
# avg_nz_pix_neighbors = Int(round(length(s.nzval)/s.n/2))
# i = 100
# plot_subsection = s[i, (i-avg_nz_pix_neighbors-5):(i+avg_nz_pix_neighbors+5)]
# plot(plot_subsection)
# plot!(iszero.(plot_subsection)./10)
