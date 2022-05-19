using LinearAlgebra
using Plots
using Statistics
import StellarSpectraObservationFitting as SSOF

_plt_dpi = 400
_plt_size = (1920,1080)
_thickness_scaling = 2
_theme = :default
# _theme = :juno
_theme == :juno ? base_color = :white : base_color=:black
_plot(; dpi = _plt_dpi, size = _plt_size, thickness_scaling=_thickness_scaling, kwargs...) =
    plot(; dpi=dpi, size=size, thickness_scaling=thickness_scaling, kwargs...)
plot_spectrum(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_rv(; xlabel = "Time (d)", ylabel = "RV (m/s)", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_scores(; xlabel = "Time (d)", ylabel = "Weights + Const", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
theme(_theme)
function _scatter(x::AbstractVecOrMat, y::AbstractVecOrMat; kwargs...)
    plt = _plot(; kwargs...)
    scatter!(plt, x, y; kwargs...)
	return plt
end
_scatter!(plt::Union{Plots.AbstractPlot,Plots.AbstractLayout}, x::AbstractVecOrMat, y::AbstractVecOrMat; markerstrokewidth::Real=0, kwargs...) = scatter!(plt, x, y; markerstrokewidth=markerstrokewidth, kwargs...)
_theme == :default ? plt_colors = palette(_theme).colors.colors : plt_colors = PlotThemes._themes[_theme].defaults[:palette].colors.colors

plot_model_rvs!(plt, times, rvs, rvs_σ; label="", xlabel="", markerstrokewidth=1, kwargs...) = _scatter!(plt, times, rvs; yerror=rvs_σ, label=label*" RVs, std: $(round(std(rvs), digits=3)), intra night std: $(round(SSOF.intra_night_std(rvs, times), digits=3))", xlabel=xlabel, markerstrokewidth=markerstrokewidth, kwargs...)
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_rv(; legend=:bottomleft, layout=grid(2, 1, heights=[0.7, 0.3]))
    ervs = inst_rvs .- median(inst_rvs)
    mrvs = model_rvs .- median(model_rvs)
    plot_model_rvs!(plt[1], inst_times, ervs, inst_rvs_σ; label="Instrument", kwargs...)
    plot_model_rvs!(plt[1], times_nu, mrvs, model_rvs_σ; label="SSOF", kwargs...)
    resids = mrvs - ervs
    _scatter!(plt[2], times_nu, resids, ylabel="SSOF - Instrument (m/s)", yerror=sqrt.(model_rvs_σ .^ 2 + inst_rvs_σ .^ 2), alpha = 0.5, label="std: $(round(std(resids), digits=3))", markerstrokewidth=1)
    if display_plt; display(plt) end
    return plt
end
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}, ccf_rvs::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_model_rvs(times_nu, model_rvs, model_rvs_σ, inst_times, inst_rvs, inst_rvs_σ; markerstrokewidth=1)
    _scatter!(plt[1], inst_times, ccf_rvs .- median(ccf_rvs); label="CCF RVs,      std: $(round(std(ccf_rvs), digits=3)), intra night std: $(round(SSOF.intra_night_std(ccf_rvs, inst_times), digits=3))", alpha = 0.7, kwargs...)
    if display_plt; display(plt) end
    return plt
end

c_ind_f(i) = ((i + 3) % 19) + 1
function plot_model(om::SSOF.OrderModel, airmasses::Vector, times_nu::Vector; display_plt::Bool=true, d::Union{SSOF.Data, Nothing}=nothing, o::Union{SSOF.Output, Nothing}=nothing, incl_χ²::Bool=true, kwargs...)
	plot_stellar = SSOF.is_time_variable(om.star)
	plot_telluric = SSOF.is_time_variable(om.tel)
	# plot the two templates if there is no time variation
	if (!plot_stellar) && (!plot_telluric)
		plt = plot_spectrum(; title="Constant Model", legend=:outerright, kwargs...)
		plot!(plt, om.tel.λ, om.tel.lm.μ; label="μₜₑₗ")
		plot!(plt, om.star.λ, om.star.lm.μ .- 0.5; label="μₛₜₐᵣ")
		if display_plt; display(plt) end
		return plt
	end

	n_plots = plot_stellar + plot_telluric
	plt = _plot(; layout=grid(2, n_plots), size=(n_plots * _plt_size[1],_plt_size[2]*1.5))
	shift_M = 0.2
	incl_χ² = incl_χ² && !isnothing(d) && !isnothing(o)
	if incl_χ²; χ²_base = SSOF._loss(o, om, d) end
	if plot_telluric
		inds = 1:size(om.tel.lm.M, 2)

		# basis plot
		plot!(plt[1, 1], om.star.λ, om.star.lm.μ; label="μₛₜₐᵣ", alpha=0.3, color=base_color, title="Telluric Model Bases", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
		plot!(plt[1, 1], om.tel.λ, om.tel.lm.μ; label="μₜₑₗ")

		_scatter!(plt[2, 1], times_nu, airmasses; label="Airmasses", color=plt_colors[1])
		hline!(plt[2, 1], [1]; label="", color=plt_colors[1], lw=3, alpha=0.4)

		shift_s = ceil(10 * maximum([std(om.tel.lm.s[inds[i], :] .* norm(om.tel.lm.M[:, inds[i]])) for i in inds])) / 2
		half_shift_s = ceil(shift_s) / 2
		holder = copy(om.tel.lm.s)
	    for i in reverse(inds)
	        c_ind = c_ind_f(i - inds[1])
			norm_M = norm(view(om.tel.lm.M, :, i))

			# basis plot
	        plot!(plt[1, 1], om.tel.λ, (view(om.tel.lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); label="Basis $i", color=plt_colors[c_ind])

			# weights plot
			if incl_χ²
				om.tel.lm.s[i, :] .= 0
				Δχ² = 1 - (χ²_base / SSOF._loss(o, om, d; tel=vec(om.tel.lm)))
				om.tel.lm.s[i, :] .= view(holder, i, :)
				_scatter!(plt[2, 1], times_nu, (view(om.tel.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i (Δχ² = $(round(Δχ²; digits=3)))", color=plt_colors[c_ind], title="Telluric Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			else
				_scatter!(plt[2, 1], times_nu, (view(om.tel.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i", color=plt_colors[c_ind], title="Telluric Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			end
			hline!(plt[2, 1], [-(shift_s * (i - 1) + half_shift_s)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
	    end
	end
	if plot_stellar
		plot_telluric ? c_offset = inds[end] - 1 : c_offset = 1
		inds = 1:size(om.star.lm.M, 2)

		# basis plot
		plot!(plt[1, n_plots], om.tel.λ, om.tel.lm.μ; label="μₜₑₗ", alpha=0.3, color=base_color, title="Stellar Model Bases", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
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
			if incl_χ²
				om.star.lm.s[i, :] .= 0
				Δχ² = 1 - (χ²_base / SSOF._loss(o, om, d; star=vec(om.star.lm)))
				om.star.lm.s[i, :] .= view(holder, i, :)
				_scatter!(plt[2, n_plots], times_nu, (view(om.star.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i (Δχ² = $(round(Δχ²; digits=3)))", color=plt_colors[c_ind], title="Stellar Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			else
				_scatter!(plt[2, n_plots], times_nu, (view(om.star.lm.s, i, :) .* norm_M) .- (shift_s * (i - 1) + half_shift_s); label="Weights $i", color=plt_colors[c_ind], title="Stellar Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
			end
			hline!(plt[2, n_plots], [-(shift_s * (i - 1) + half_shift_s)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
		end
	end

    if display_plt; display(plt) end
    return plt
end
plot_model(mws::SSOF.ModelWorkspace, airmasses::Vector, times_nu::Vector; kwargs...) =
	plot_model(mws.om, airmasses, times_nu; d=mws.d, o=mws.o, kwargs...)

function status_plot(mws::SSOF.ModelWorkspace; tracker::Int=0, display_plt::Bool=true, include_χ²::Bool=true, kwargs...)
    o = mws.o
	d = mws.d
	time_average(a) = mean(a; dims=2)
	obs_mask = vec(all(.!(isinf.(d.var)); dims=2))
    obs_λ = time_average(exp.(d.log_λ_obs))
    plot_star_λs = time_average(exp.(d.log_λ_star))
	include_χ² ?
		plt = plot_spectrum(; legend = :bottomright, layout = grid(3, 1, heights=[0.6, 0.2, 0.2]), ylabel="Flux + Constant Shift", kwargs...) :
		plt = plot_spectrum(; legend = :bottomright, layout = grid(2, 1, heights=[0.85, 0.15]), kwargs...)

	tel_model = time_average(o.tel)
    plot!(plt[1], obs_λ, tel_model, label="Mean Telluric Model")

    shift = 1.1 - minimum(tel_model)

	typeof(mws.om) <: OrderModelWobble ?
		star_model = time_average(o.star) :
		star_model = time_average(o.star + o.rv)

    plot!(plt[1], obs_λ, star_model .- shift, label="Mean Stellar Model")

    shift += 1.1 - minimum(star_model)
    plot!(plt[1], obs_λ, time_average(o.total) .- shift, label="Mean Full Model", color=base_color)

    _scatter!(plt[2], obs_λ[obs_mask], time_average(abs.(view(d.flux, obs_mask, :) - view(o.total, obs_mask, :))), ylabel="MAD", label="", alpha=0.5, color=base_color, xlabel="", ms=1.5)

	if include_χ²
		_scatter!(plt[3], obs_λ, -sum(SSOF._loss_diagnostic(mws); dims=2), ylabel="Remaining χ²", label="", alpha=0.5, color=base_color, xlabel="", ms=1.5)
	end

    if display_plt; display(plt) end
    return plt
end

function component_test_plot(ys::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector; size=(_plt_size[1],_plt_size[2]*1.5), ylabel="ℓ")
    plt = _plot(; ylabel=ylabel, layout=grid(2, 1), size=size)
	lims = [maximum(ys[2:end, 2:end]), minimum(ys[2:end, 2:end])]
	ylabel=="ℓ" ? lims[1] = maximum(ys) : lims[2] = minimum(ys)
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

function save_model_plots(mws, airmasses, times_nu, save_path::String; display_plt::Bool=true, incl_χ²::Bool=true, kwargs...)
	plt = plot_model(mws, airmasses, times_nu; display_plt=display_plt, incl_χ²=incl_χ², kwargs...);
	png(plt, save_path * "model.png")

	plt = status_plot(mws; display_plt=display_plt, kwargs...);
	png(plt, save_path * "status_plot.png")
end
