using LinearAlgebra
using Plots

_plt_dpi = 400
_plt_size = (750, 500)
plot_spectrum(; kwargs...) = plot(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux", dpi = _plt_dpi, size = _plt_size, kwargs...)
plot_rv(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "RV (m/s)", dpi = _plt_dpi, size = _plt_size, kwargs...)
plot_scores(; kwargs...) = plot(; xlabel = "Time (d)", ylabel = "Weights", dpi = _plt_dpi, size = _plt_size, kwargs...)
plt_colors = palette(:default).colors.colors

function plot_model_rvs(times_nu::AbstractVector{T}, rvs_naive::AbstractVector{T}, rvs_notel::AbstractVecOrMat{T}, rvs_notel_opt::AbstractVecOrMat{T}) where {T<:Real}
    predict_plot = plot_rv()
    plot!(predict_plot, times_nu, rvs_naive, st=:scatter, ms=3, color=:red, label="Naive, std: $(round(std(rvs_naive), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel, st=:scatter, ms=3, color=:lightgreen, label="Before optimization, std: $(round(std(rvs_notel), digits=3))")
    plot!(predict_plot, times_nu, rvs_notel_opt, st=:scatter, ms=3, color=:darkgreen, label="After optimization, std: $(round(std(rvs_notel_opt), digits=3))")
    display(predict_plot)
    return predict_plot
end

function plot_stellar_model_bases(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.star.lm.M, 2))
    predict_plot = plot_spectrum(; title="Stellar model bases", legend=:outerright)
    plot!(tfom.star.λ, tfom.star.lm.μ; label="μ")
    shift = 0.2
    for i in reverse(inds)
        plot!(tfom.star.λ, (tfom.star.lm.M[:, i] ./ norm(tfom.star.lm.M[:, i])) .- shift * (i - 1); label="basis $i", color=plt_colors[i - inds[1] + 2])
    end
    display(predict_plot)
    return predict_plot
end
function plot_stellar_model_scores(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.star.lm.M, 2))
    predict_plot = plot_scores(; title="Stellar model scores", legend=:outerright)
    shift = 5 * maximum([std(tfom.star.lm.s[inds[i], :] .* norm(tfom.star.lm.M[:, inds[i]])) for i in inds])
    for i in reverse(inds)
        scatter!(times_nu, (tfom.star.lm.s[i, :] .* norm(tfom.star.lm.M[:, i])) .- shift * (i - 1); label="weights $i", color=plt_colors[i - inds[1] + 2])
        hline!([-shift * (i - 1)]; label="", color=plt_colors[i - inds[1] + 2], lw=3, alpha=0.4)
    end
    display(predict_plot)
    return predict_plot
end

function plot_telluric_model_bases(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.tel.lm.M, 2))
    predict_plot = plot_spectrum(; title="Telluric model bases", legend=:outerright)
    plot!(tfom.tel.λ, tfom.tel.lm.μ; label="μ")
    shift = 0.2
    for i in reverse(inds)
        plot!(tfom.tel.λ, (tfom.tel.lm.M[:, i] ./ norm(tfom.tel.lm.M[:, i])) .- shift * (i - 1); label="basis $i", color=plt_colors[i - inds[1] + 2])
    end
    display(predict_plot)
    return predict_plot
end
function plot_telluric_model_scores(tfom::StellarSpectraObservationFitting.TFOrderModel; inds::UnitRange=1:size(tfom.tel.lm.M, 2))
    predict_plot = plot_scores(; title="Telluric model scores", legend=:outerright)
    scatter!(times_nu, airmasses; label="airmasses")
    hline!([1]; label="", color=plt_colors[1], lw=3, alpha=0.4)
    shift = 5 * maximum([std(tfom.tel.lm.s[inds[i], :] .* norm(tfom.tel.lm.M[:, inds[i]])) for i in inds])
    for i in reverse(inds)
        scatter!(times_nu, (tfom.tel.lm.s[i, :] .* norm(tfom.tel.lm.M[:, i])) .- shift * (i - 1); label="weights $i", color=plt_colors[i - inds[1] + 2])
        hline!([-shift * (i - 1)]; label="", color=plt_colors[i - inds[1] + 2], lw=3, alpha=0.4)
    end
    display(predict_plot)
    return predict_plot
end
