# using Pkg
# Pkg.activate("EXPRES")

using Distributions
using LinearAlgebra
using Plots
using QuadGK

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
gxgpdf(x; μ1=0, σ1=1, μ2=0, σ2=1) = gxgpdf(Normal(μ1, σ1), Normal(μ2, σ2), x)

@assert gxgpdf(Normal(1, 2), Normal(3, 4), 5) == pdf(Normal(1, 2), 5) * pdf(Normal(3, 4), 5)

# Gaussian CDF
gcdf(μ::Real, σ::Real, x) = gcdf(Normal(μ, σ), x)
gcdf(nd::Normal, x) = cdf.(nd, x)

# Integral over a pixel
gpdf_pix_int(nd::Normal, pix::Int) = gcdf(nd, pix + 1 / 2) - gcdf(nd, pix - 1 / 2)

## Should we use the LSF at center or a true LSF integrated over a pixel width?
# Yes, esp if we apply an area correction. Then we end up with a constant, slightly sharper profile

test_FWHM = 4.5
test_σ = test_FWHM / (2 * sqrt(2 * log(2)))

x = LinRange(-5,5,1001)
x[501]
μs = 0:1/2:10
i = 1
for i in 1:2:18#length(μs)
    y1 = (gcdf.(Normal(μs[i]+1/2, test_σ), x) - gcdf.(Normal(μs[i]-1/2, test_σ), x)) .* (gcdf.(Normal(-μs[i]+1/2, test_σ), x) - gcdf.(Normal(-μs[i]-1/2, test_σ), x))
    c, nd = gxgpdf(Normal(μs[i], test_σ), Normal(-μs[i], test_σ))
    y2 = c .* gpdf.(nd, x)
    areadif = (sum(y1) / sum(y2)) - 1
    plt = plot(x, y1; label="Exact", title="$(Int(2*μs[i])) pix separation")
    plot!(x, y2; label="Approx")
    y2 .*= 1 + areadif
    plot!(x, y2; label="Approx, corrected", c=plt_colors[6])
    png(plt, "lsf_$(Int(2*μs[i]))_pix_sep")
    maxdifrat = (y2[501] - y1[501]) / y1[501]
    println("pix sep: $(i-1), std: $(std(y1-y2)), maxdifrat: $maxdifrat, areadif: $areadif")
end
i = 20
@time y1 = (gcdf.(Normal(μs[i]+1/2, test_σ), x) - gcdf.(Normal(μs[i]-1/2, test_σ), x)) .* (gcdf.(Normal(-μs[i]+1/2, test_σ), x) - gcdf.(Normal(-μs[i]-1/2, test_σ), x))
@time begin
    c, nd = gxgpdf(Normal(μs[i], test_σ), Normal(-μs[i], test_σ))
    y2 = c .* gpdf.(nd, x)
end
areadif = (sum(y1) / sum(y2)) - 1
y2 .*= 1 + areadif
plt = plot(x, [y1, y2])
maxdifrat = (y2[501] - y1[501]) / y1[501]

# functions to get a better approximation for what the area should be, had we actually integrated the LSF over pixels
f(x, pix_sepx2) = (gcdf.(Normal(pix_sepx2+1/2, test_σ), x) - gcdf.(Normal(pix_sepx2-1/2, test_σ), x)) .* (gcdf.(Normal(-pix_sepx2+1/2, test_σ), x) - gcdf.(Normal(-pix_sepx2-1/2, test_σ), x))
function f2(pix_sepx2)
    integral, _ = quadgk(x -> f(x, pix_sepx2), -100, 100, rtol=1e-8)
    return integral
end
function f3(pix_sepx2)
    c, _ = gxgpdf(Normal(pix_sepx2, test_σ), Normal(-pix_sepx2, test_σ))
    return c
end
[f3(μ) for μ in μs]
[f2(μ) for μ in μs] ./ [f3(μ) for μ in μs]
8 / test_σ
n_pix = 100
n_rep = 500
noise = zeros(n_pix)
test_FWHM = 4.5
test_σ = test_FWHM / (2 * sqrt(2 * log(2)))
ev = zeros(n_pix, n_pix)



lsf_areas = zeros(n_pix)
for i in 1:Int(round(10 * test_σ))
    lsf_areas[i] = f2((i-1)/2)
end

@progress for k in 1:n_rep
    noise[:] = randn(n_pix)
    for i in 1:n_pix
        for j in 1:i
            _, comb_lsf = gxgpdf(; μ1=i-1/2, σ1=test_σ, μ2=j-1/2, σ2=test_σ)
            ev[j, i] += lsf_areas[i - j + 1] * sum([gpdf_pix_int(comb_lsf, l) for l in 1:n_pix] .* (noise .^ 2))
        end
    end
end
ev ./= n_rep
ev = Symmetric(ev)
heatmap(ev)
histogram(vec(sum(ev;dims=1)))

## Finding LSF width as a function of λ

using CSV, DataFrames
SSOF_path = dirname(dirname(pathof(SSOF)))
if interactive
    include(SSOF_path * "/src/_plot_functions.jl")
end

eo = CSV.read("D:/Christian/Downloads/expres_psf.txt", DataFrame)
filter!(:line => ==("LFC"), eo)
sort!(eo, ["wavenumber [1/cm]"])

λs = 1e8 ./ eo."wavenumber [1/cm]"
FWHM = λs .* (eo."fwhm [1/cm]" ./ eo."wavenumber [1/cm]")
plt = my_scatter(λs, FWHM; label="", xlabel="Wavelength (Å)", ylabel="LSF FWHM (Å?)")
png(plt, "expres_lsf")

λs = eo."wavenumber [1/cm]"[:]
FWHM = eo."fwhm [1/cm]"[:]
inds1 = eo.order .== 38
inds2 = eo.order .== 39
plt = my_scatter(λs[inds1], FWHM[inds1]; label="", xlabel="Wavelength (1/cm)", ylabel="LSF FWHM (1/cm)")
scatter!(λs[inds2], FWHM[inds2]; label="")
png(plt, "expres_lsf_zoom")

poly_order = 2
dm = ones(size(eo, 1), 2 + poly_order)
orders = [i for i in 1:100 if i in eo.order]
for order in orders
    inds_temp = eo.order .== order
    df = filter(:order => ==(order), eo)
    dm[inds_temp, 2] = df."wavenumber [1/cm]"
    dm[inds_temp, 3] = dm[inds_temp, 2] .- mean(dm[inds_temp, 2])
    for i in 2:poly_order
        dm[inds_temp, i+2] = dm[inds_temp, 3] .^ i
    end
end
w = SSOF.general_lst_sq(dm, FWHM, (eo."fwhm error [1/cm]") .^ 2)
model = dm * w
plt = my_scatter(λs, FWHM; label="", xlabel="Wavelength (1/cm)", ylabel="LSF FWHM (1/cm)")
for order in orders
    inds_temp = eo.order .== order
    plot!(λs[inds_temp], model[inds_temp]; label="", lw=4)
end
png(plt, "expres_lsf_model")

plt = my_scatter(λs[inds1], FWHM[inds1]; label="", xlabel="Wavelength (1/cm)", ylabel="LSF FWHM (1/cm)")
scatter!(λs[inds2], FWHM[inds2]; label="")
plot!(λs[inds1], model[inds1]; label="model", lw=6, c=plt_colors[6])
plot!(λs[inds2], model[inds2]; label="model", lw=6)
png(plt, "expres_lsf_model_zoom")

resids = FWHM ./ model
std(resids)
plt = my_scatter(λs, resids; label="data / model", xlabel="Wavelength (1/cm)", ylabel="LSF FWHM (1/cm)")
png(plt, "expres_lsf_resids")
plt = scatter(λs[inds1], resids[inds1]; label="data / model", xlabel="Wavelength (1/cm)", ylabel="LSF FWHM (1/cm)")
scatter!(λs[inds2], resids[inds2]; label="data / model")
png(plt, "expres_lsf_resids_zoom")
