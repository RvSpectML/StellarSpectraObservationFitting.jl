using LinearAlgebra
using Plots
using Statistics
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

_plt_dpi = 400
_plt_size = (1920,1080)
_thickness_scaling = 2
# _theme = :default
_theme = :juno
_my_plot(; dpi = _plt_dpi, size = _plt_size, thickness_scaling=_thickness_scaling, kwargs...) =
    plot(; dpi=dpi, size=size, thickness_scaling=thickness_scaling, kwargs...)
function my_plot(x, y; kwargs...)
    plt = _my_plot(; kwargs...)
    plot!(plt, x, y; kwargs...)
end
plot_spectrum(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_rv(; xlabel = "Time (d)", ylabel = "RV (m/s)", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_scores(; xlabel = "Time (d)", ylabel = "Weights + Const", kwargs...) =
    _my_plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
theme(_theme)
function my_scatter(x::AbstractVecOrMat, y::AbstractVecOrMat; kwargs...)
    plt = _my_plot(; kwargs...)
    scatter!(plt, x, y; kwargs...)
end
my_scatter!(plt::Union{Plots.AbstractPlot,Plots.AbstractLayout}, x::AbstractVecOrMat, y::AbstractVecOrMat; markerstrokewidth::Real=0, kwargs...) = scatter!(plt, x, y; markerstrokewidth=markerstrokewidth, kwargs...)
_theme == :default ? plt_colors = palette(_theme).colors.colors : plt_colors = PlotThemes._themes[_theme].defaults[:palette].colors.colors
intra_night_std(rvs, times) = median([std(rvs[i]) for i in SSOF.observation_night_inds(times) if length(i)>3])

plot_model_rvs!(plt, times, rvs, rvs_σ; label="", xlabel="", markerstrokewidth=1, kwargs...) = my_scatter!(plt, times, rvs; yerror=rvs_σ, label=label*" RVs, std: $(round(std(rvs), digits=3)), intra night std: $(round(intra_night_std(rvs, times), digits=3))", xlabel=xlabel, markerstrokewidth=markerstrokewidth, kwargs...)
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_rv(; legend=:bottomleft, layout=grid(2, 1, heights=[0.7, 0.3]))
    ervs = inst_rvs .- median(inst_rvs)
    mrvs = model_rvs .- median(model_rvs)
    plot_model_rvs!(plt[1], inst_times, ervs, inst_rvs_σ; label="Instrument", kwargs...)
    plot_model_rvs!(plt[1], times_nu, mrvs, model_rvs_σ; label="Model", kwargs...)
    resids = mrvs - ervs
    my_scatter!(plt[2], times_nu, resids, ylabel="model - Instrument (m/s)", yerror=sqrt.(model_rvs_σ .^ 2 + inst_rvs_σ .^ 2), alpha = 0.5, label="std: $(round(std(resids), digits=3))", markerstrokewidth=1)
    if display_plt; display(plt) end
    return plt
end
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}, ccf_rvs::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_model_rvs(times_nu, model_rvs, model_rvs_σ, inst_times, inst_rvs, inst_rvs_σ; markerstrokewidth=1)
    my_scatter!(plt[1], inst_times, ccf_rvs .- median(ccf_rvs); label="CCF RVs,      std: $(round(std(ccf_rvs), digits=3)), intra night std: $(round(intra_night_std(ccf_rvs, inst_times), digits=3))", alpha = 0.7, kwargs...)
    if display_plt; display(plt) end
    return plt
end

c_ind_f(i) = ((i + 3) % 19) + 1
function plot_model(mws::SSOF.ModelWorkspace; display_plt::Bool=true, kwargs...)
    om = mws.om
	plot_stellar = !(typeof(om.star.lm) <: SSOF.TemplateModel)
	plot_telluric = !(typeof(om.tel.lm) <: SSOF.TemplateModel)
	# plot the two templates if there is no time variation
	if (!plot_stellar) && (!plot_telluric)
		plt = plot_spectrum(; title="Telluric Model Bases", legend=:outerright, kwargs...)
		plot!(plt, om.tel.λ, om.tel.lm.μ; label="μₜₑₗ")
		plot!(plt, om.star.λ, om.star.lm.μ .- 0.5; label="μₛₜₐᵣ")
		if display_plt; display(plt) end
		return plt
	end

	n_plots = plot_stellar + plot_telluric
	plt = _my_plot(; layout=grid(2, n_plots), size=(n_plots * _plt_size[1],_plt_size[2]*1.5))
	shift_M = 0.2
	χ²_base = SSOF._loss(mws)
	if plot_telluric
		inds = 1:size(mws.om.tel.lm.M, 2)

		# basis plot
		plot!(plt[1, 1], om.star.λ, om.star.lm.μ; label="μₛₜₐᵣ", alpha=0.3, color=:white, title="Telluric Model Bases", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
		plot!(plt[1, 1], om.tel.λ, om.tel.lm.μ; label="μₜₑₗ")

		shift_s = ceil(10 * maximum([std(om.tel.lm.s[inds[i], :] .* norm(om.tel.lm.M[:, inds[i]])) for i in inds])) / 2
		half_shift_s = ceil(shift_s) / 2
		holder = copy(om.tel.lm.s)
	    for i in reverse(inds)
	        c_ind = c_ind_f(i - inds[1])
			norm_M = norm(view(om.tel.lm.M, :, i))

			# basis plot
	        plot!(plt[1, 1], om.tel.λ, (view(om.tel.lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); label="Basis $i", color=plt_colors[c_ind])

			# weights plot
			om.tel.lm.s[i, :] .= 0
			Δχ² = 1 - (χ²_base / SSOF._loss(mws; tel=vec(om.tel.lm)))
			om.tel.lm.s[i, :] .= view(holder, i, :)
			my_scatter!(plt[2, 1], times_nu, (view(om.tel.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i (Δχ² = $(round(Δχ²; digits=3)))", color=plt_colors[c_ind], title="Telluric Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			hline!(plt[2, 1], [-(shift_s * (i - 1) + half_shift_s)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
	    end
	end
	if plot_stellar
		plot_telluric ? c_offset = inds[end] - 1 : c_offset = 1
		inds = 1:size(mws.om.star.lm.M, 2)

		# basis plot
		plot!(plt[1, n_plots], om.tel.λ, om.tel.lm.μ; label="μₜₑₗ", alpha=0.3, color=:white, title="Stellar Model Bases", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
		plot!(plt[1, n_plots], om.star.λ, om.star.lm.μ; label="μₛₜₐᵣ")

		shift_s = ceil(10 * maximum([std(om.star.lm.s[inds[i], :] .* norm(om.star.lm.M[:, inds[i]])) for i in inds])) / 2
		half_shift_s = ceil(shift_s) / 2
		holder = copy(om.star.lm.s)
		for i in reverse(inds)
			c_ind = c_ind_f(i + c_offset)
			norm_M = norm(view(om.star.lm.M, :, i))

			# basis plot
			plot!(plt[1, n_plots], om.star.λ, (view(om.star.lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); label="Basis $i", color=plt_colors[c_ind])

			# weights plot
			om.star.lm.s[i, :] .= 0
			Δχ² = 1 - (χ²_base / SSOF._loss(mws; star=vec(om.star.lm)))
			om.star.lm.s[i, :] .= view(holder, i, :)
			my_scatter!(plt[2, n_plots], times_nu, (view(om.star.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i (Δχ² = $(round(Δχ²; digits=3)))", color=plt_colors[c_ind], title="Stellar Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			hline!(plt[2, n_plots], [-(shift_s * (i - 1) + half_shift_s)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
		end
	end

    if display_plt; display(plt) end
    return plt
end

function status_plot(o::SSOF.Output, d::SSOF.Data; plot_epoch::Int=10, tracker::Int=0, display_plt::Bool=true, kwargs...)
    obs_mask = .!(isinf.(d.var[:, plot_epoch]))
    obs_λ = exp.(d.log_λ_obs[:, plot_epoch])
    plot_star_λs = exp.(d.log_λ_star[:, plot_epoch])
    plt = plot_spectrum(; legend = :bottomright, layout = grid(2, 1, heights=[0.85, 0.15]), kwargs...)

    plot!(plt[1], obs_λ, o.tel[:, plot_epoch], label="Telluric Model")

    shift = 1.1 - minimum(o.tel[:, plot_epoch])
    star_model = o.star[:, plot_epoch] + o.rv[:, plot_epoch]
    plot!(plt[1], obs_λ, star_model .- shift, label="Stellar Model")

    shift += 1.1 - minimum(star_model)
    my_scatter!(plt[1], obs_λ[obs_mask], d.flux[obs_mask, plot_epoch] .- shift, label="Observed Data", color=:white, alpha=0.1, xlabel="")
    plot!(plt[1], obs_λ, o.total[:, plot_epoch] .- shift, label="Full Model", ls=:dash, color=:white)
    # plot!(plt[1], obs_λ, o.tel[:, plot_epoch] .* star_model .- shift, label="Full Model", ls=:dash, color=:white)

    my_scatter!(plt[2], obs_λ[obs_mask], d.flux[obs_mask, plot_epoch] - o.total[obs_mask, plot_epoch], ylabel="Residuals", label="", alpha=0.1, color=:white)
    # my_scatter!(plt[2], obs_λ, d.flux[:, plot_epoch] - (o.tel[:, plot_epoch] .* star_model), ylabel="Residuals", label="", alpha=0.1, color=:white)
    if display_plt; display(plt) end
    return plt
end
status_plot(mws::SSOF.ModelWorkspace; kwargs...) =
    status_plot(mws.o, mws.d; kwargs...)

function component_test_plot(ys::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector; size=(_plt_size[1],_plt_size[2]*1.5), ylabel="ℓ")
    plt = _my_plot(; ylabel=ylabel, layout=grid(2, 1), size=size)
	lims = [maximum(ys[2:end, 2:end]), minimum(ys[2:end, 2:end])]
	buffer = 0.5 * (lims[1] - lims[2])
	ylims!(plt, lims[2] - buffer, lims[1] + buffer)
    for i in eachindex(test_n_comp_tel)
        plot!(plt[1], test_n_comp_star, ys[i, :]; label="$(test_n_comp_tel[i]) tel", xlabel="# of stellar components")
    end
    for i in eachindex(test_n_comp_star)
        plot!(plt[2], test_n_comp_tel, ys[:, i]; label="$(test_n_comp_star[i]) stellar", xlabel="# of telluric components")
    end
    display(plt)
    return plt
end

function model_choice_plots(ℓ, aics, bics, test_n_comp_tel, test_n_comp_star, save_path::String)
	plt = component_test_plot(ℓ, test_n_comp_tel, test_n_comp_star);
	png(plt, save_path * "l_plot.png")

	plt = component_test_plot(aics, test_n_comp_tel, test_n_comp_star; ylabel="AIC");
	png(plt, save_path * "aic_plot.png")

	plt = component_test_plot(bics, test_n_comp_tel, test_n_comp_star; ylabel="BIC");
	png(plt, save_path * "bic_plot.png")
end
