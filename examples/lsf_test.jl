# using Pkg
# Pkg.activate("EXPRES")

using Distributions
using LinearAlgebra
using Plots

# Gaussian PDFs
gpdf(μ::Real, σ::Real, x) = gpdf(Normal(μ, σ), x)
gpdf(nd::Normal, x) = pdf.(nd, x)

# Gaussian * Gaussian PDFs
function gxgpdf(nd1::Normal, nd2::Normal)
    γ = nd1.σ ^ 2 + nd2.σ ^ 2
    c = gpdf(nd1.μ, sqrt(γ), nd2.μ)
    σ² =  nd1.σ ^ 2 * nd2.σ ^ 2 / γ
    μ = (nd2.μ * nd1.σ ^ 2 + nd1.μ * nd2.σ ^ 2) / γ
    return c, Normal(μ, sqrt(σ²))
end
gxgpdf(; μ1=0, σ1=1, μ2=0, σ2=1) = gxgpdf(Normal(μ1, σ1), Normal(μ2, σ2))
function gxgpdf(nd1::Normal, nd2::Normal, x)
    c, nd = gxgpdf(nd1, nd2)
    return c * gpdf(nd, x)
end

@assert gxgpdf(Normal(1, 2), Normal(3, 4), 5) == pdf(Normal(1, 2), 5) * pdf(Normal(3, 4), 5)

# Gaussian CDF
gcdf(μ::Real, σ::Real, x) = gcdf(Normal(μ, σ), x)
gcdf(nd::Normal, x) = cdf.(nd, x)

# Integral over a pixel
gpdf_pix_int(nd::Normal, pix::Int) = gcdf(nd, pix + 1 / 2) - gcdf(nd, pix - 1 / 2)

n_pix = 50
n_rep = 100
noise = zeros(n_pix)
test_FWHM = 4.5
test_σ = test_FWHM / (2 * sqrt(2 * log(2)))
ev = zeros(n_pix, n_pix)
@progress for k in 1:n_rep
    noise[:] = randn(n_pix)
    for i in 1:n_pix
        for j in 1:i
            c, comb_lsf = gxgpdf(; μ1=i-1/2, σ1=test_σ, μ2=j-1/2, σ2=test_σ)
            ev[j, i] += c * sum([gpdf_pix_int(comb_lsf, l) for l in 1:n_pix] .* (noise .^ 2))
        end
    end
end
ev ./= n_rep
ev = Symmetric(ev)

heatmap(ev)
