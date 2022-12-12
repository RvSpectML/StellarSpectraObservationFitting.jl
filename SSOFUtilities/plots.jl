using LinearAlgebra
using Plots
using Statistics
import StellarSpectraObservationFitting as SSOF
using LaTeXStrings

_plt_dpi = 400
_plt_size = (1920,1080)
_thickness_scaling = 2
_theme = :default
# _theme = :juno
if _theme == :juno
	base_color = :white
	anti_color = :black
else
	base_color = :black
	anti_color = :white
end
_plot(; dpi = _plt_dpi, size = _plt_size, thickness_scaling=_thickness_scaling, kwargs...) =
    plot(; dpi=dpi, size=size, thickness_scaling=thickness_scaling, kwargs...)
plot_spectrum(; xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_rv(; xlabel = "Time (d)", ylabel = "RV (m/s)", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
plot_scores(; xlabel = "Time (d)", ylabel = "Weights + Const", kwargs...) =
    _plot(; xlabel=xlabel, ylabel=ylabel, kwargs...)
theme(_theme)
_scatter!(plt::Union{Plots.AbstractPlot,Plots.AbstractLayout}, x::AbstractVecOrMat, y::AbstractVecOrMat; markerstrokewidth::Real=0, kwargs...) = scatter!(plt, x, y; markerstrokewidth=markerstrokewidth, kwargs...)
_theme == :default ? plt_colors = palette(_theme).colors.colors : plt_colors = PlotThemes._themes[_theme].defaults[:palette].colors.colors

function rv_legend(label, rvs, times)
	intra_night = SSOF.intra_night_std(rvs, times; show_warn=false)
	isinf(intra_night) ? appen = "" : appen = ", intra night std: $(round(intra_night, digits=3))"
	return label * " RVs, std: $(round(std(rvs), digits=3))" * appen
end
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}; display_plt::Bool=true, inst_str::String="Instrument", msw::Real=0.5, alpha=0.7, kwargs...) where {T<:Real}
    plt = plot_rv(; legend=:bottomright, layout=grid(2, 1, heights=[0.7, 0.3]))
    ervs = inst_rvs .- median(inst_rvs)
    mrvs = model_rvs .- median(model_rvs)
	scatter!(plt[1], inst_times, ervs; yerror=inst_rvs_σ, msc=0.4*plt_colors[1], label=inst_str * " (std: $(round(std(ervs), digits=3)), σ: $(round(mean(inst_rvs_σ), digits=3)))", alpha = alpha, msw=msw, kwargs...)
	scatter!(plt[1], times_nu, mrvs; yerror=model_rvs_σ, msc=0.4*plt_colors[2], label="SSOF (std: $(round(std(mrvs), digits=3)), σ: $(round(mean(model_rvs_σ), digits=3)))", alpha = alpha, msw=msw, kwargs...)

    resids = mrvs - ervs
    scatter!(plt[2], times_nu, resids; c=:black, ylabel="SSOF - " * inst_str * " (m/s)", yerror=sqrt.(model_rvs_σ .^ 2 + inst_rvs_σ .^ 2), alpha = alpha, msw=msw, label="std: $(round(std(resids), digits=3))")
    if display_plt; display(plt) end
    return plt
end
function plot_model_rvs(times_nu::AbstractVector{T}, model_rvs::AbstractVecOrMat{T}, model_rvs_σ::AbstractVecOrMat{T}, inst_times::AbstractVector{T}, inst_rvs::AbstractVector{T}, inst_rvs_σ::AbstractVector{T}, ccf_rvs::AbstractVector{T}; display_plt::Bool=true, kwargs...) where {T<:Real}
    plt = plot_model_rvs(times_nu, model_rvs, model_rvs_σ, inst_times, inst_rvs, inst_rvs_σ)
    _scatter!(plt[1], inst_times, ccf_rvs .- median(ccf_rvs); label=rv_legend("CCF", ccf_rvs, inst_times), alpha = 0.7, markerstrokewidth=0.5, kwargs...)
    if display_plt; display(plt) end
    return plt
end

function plot_stellar_with_lsf!(plt, om::SSOF.OrderModel, y::AbstractVector; d::Union{SSOF.Data, Nothing}=nothing, alpha=1, label="", kwargs...)
	if typeof(d) <: SSOF.LSFData
		plot!(plt, om.star.λ, y; alpha=alpha/2, label="", kwargs...)
		typeof(om) <: SSOF.OrderModelWobble ?
			y2 = d.lsf * SSOF.spectra_interp(y, om.rv[1] + om.bary_rvs[1], om.b2o; sih_ind=1) :
			y2 = d.lsf * SSOF.spectra_interp(y, om.b2o[1])
		plot!(plt, exp.(d.log_λ_star[:, 1]), y2; alpha=alpha, label=label, kwargs...)
	else
		plot!(plt, om.star.λ, y; alpha=alpha, label=label, kwargs...)
	end
end
function plot_telluric_with_lsf!(plt, om::SSOF.OrderModel, y::AbstractVector; d::Union{SSOF.Data, Nothing}=nothing, alpha=1, label="", kwargs...)
	if typeof(d) <: SSOF.LSFData
		plot!(plt, om.tel.λ, y; alpha=alpha/2, label="", kwargs...)
		plot!(plt, exp.(d.log_λ_obs[:, 1]), d.lsf * SSOF.spectra_interp(y, om.t2o[1]); alpha=alpha, label=label, kwargs...)
	else
		plot!(plt, om.tel.λ, y; alpha=alpha, label=label, kwargs...)
	end
end
function plot_telluric_with_lsf!(plt, om::SSOF.OrderModel, y::AbstractMatrix; d::Union{SSOF.Data, Nothing}=nothing, alpha=1, label="", kwargs...)
	if typeof(d) <: SSOF.LSFData
		plot!(plt, om.tel.λ, vec(time_average(y)); alpha=alpha/2, label="", kwargs...)
		plot!(plt, vec(time_average(exp.(d.log_λ_obs))), vec(time_average(d.lsf * SSOF.spectra_interp(y, om.t2o))); alpha=alpha, label=label, kwargs...)
	else
		plot!(plt, om.tel.λ, vec(time_average(y)); alpha=alpha, label=label, kwargs...)
	end
end
function plot_stellar_with_lsf!(plt, om::SSOF.OrderModel, y::AbstractMatrix; d::Union{SSOF.Data, Nothing}=nothing, alpha=1, label="", kwargs...)
	if typeof(d) <: SSOF.LSFData
		plot!(plt, om.star.λ, vec(time_average(y)); alpha=alpha/2, label="", kwargs...)
		typeof(om) <: SSOF.OrderModelWobble ?
			y2 = vec(time_average(d.lsf * SSOF.spectra_interp(y, om.rv + om.bary_rvs, om.b2o))) :
			y2 = vec(time_average(d.lsf * SSOF.spectra_interp(y, om.b2o)))
		plot!(plt, vec(time_average(exp.(d.log_λ_star))), y2; alpha=alpha, label=label, kwargs...)
	else
		plot!(plt, om.star.λ, vec(time_average(y)); alpha=alpha, label=label, kwargs...)
	end
end

c_ind_f(i) = ((i + 1) % 16) + 1
function plot_model(om::SSOF.OrderModel, airmasses::Vector, times_nu::Vector; display_plt::Bool=true, d::Union{SSOF.Data, Nothing}=nothing, o::Union{SSOF.Output, Nothing}=nothing, incl_χ²::Bool=true, tel_errors::Union{AbstractMatrix, Nothing}=nothing, star_errors::Union{AbstractMatrix, Nothing}=nothing, df_act::Dict=Dict(), kwargs...)
	plot_stellar = SSOF.is_time_variable(om.star)
	plot_telluric = SSOF.is_time_variable(om.tel)
	# plot the two templates if there is no time variation
	if (!plot_stellar) && (!plot_telluric)
		plt = plot_spectrum(; title="Constant Model", legend=:outerright, kwargs...)
		plot_telluric_with_lsf!(plt, om, om.tel.lm.μ; d=d, color=plt_colors[1], label=L"\mu_\oplus")
		plot_stellar_with_lsf!(plt, om, om.star.lm.μ .- 0.5; d=d, color=plt_colors[2], label=L"\mu_\star")
		if display_plt; display(plt) end
		return plt
	end

	n_plots = plot_stellar + plot_telluric
	plt = _plot(; layout=grid(2, n_plots), size=(n_plots * _plt_size[1],_plt_size[2]*2), kwargs...)
	shift_M = 0.2
	incl_χ² = incl_χ² && !isnothing(d) && !isnothing(o)
	if incl_χ²; χ²_base = SSOF._loss(o, om, d) end

	function scaler_string(scaler::Real)
		if isapprox(1, scaler); return "" end
		return " " * string(round(scaler;digits=2)) * "x scaled"
	end

	if plot_telluric
		inds = axes(om.tel.lm.M, 2)

		# basis plot
		plot_stellar_with_lsf!(plt[1, 1], om, om.star.lm.μ; d=d, color=base_color, alpha=0.3, label=L"\mu_\star", title="Telluric Model Feature Vectors", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
		plot_telluric_with_lsf!(plt[1, 1], om, om.tel.lm.μ; d=d, color=plt_colors[1], label=L"\mu_\oplus")

		norm_Ms = [norm(view(om.tel.lm.M, :, i)) for i in inds]
		s_std = [std(view(om.tel.lm.s, inds[j], :) .* norm_Ms[j]) for j in eachindex(inds)]
		max_s_std = maximum(s_std)
		shift_s = ceil(10 * max_s_std) / 2
		_scatter!(plt[2, 1], times_nu, ((max_s_std / std(airmasses)) .* (airmasses .- mean(airmasses))) .+ shift_s; label="Airmasses (scaled)", color=base_color)
		hline!(plt[2, 1], [shift_s]; label="", color=base_color, lw=3, alpha=0.4)

		# half_shift_s = ceil(shift_s) / 2
		if incl_χ²; holder = copy(om.tel.lm.s) end
	    # for i in reverse(inds)
		for j in eachindex(inds)
			i = inds[j]
	        c_ind = c_ind_f(i - inds[1])
			norm_M = norm_Ms[j]

			# basis plot
			plot_telluric_with_lsf!(plt[1, 1], om, (view(om.tel.lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); d=d, label="Basis $i", color=plt_colors[c_ind])

			scaler = max_s_std / std(view(om.tel.lm.s, i, :) .* norm_M)
			_label = "Weights $i" * scaler_string(scaler)
			# weights plot
			if incl_χ²
				om.tel.lm.s[i, :] .= 0
				Δχ² = 1 - (χ²_base / SSOF._loss(o, om, d; tel=vec(om.tel.lm)))
				om.tel.lm.s[i, :] .= view(holder, i, :)
				_label *= " (Δχ² = $(round(Δχ²; digits=3)))"
			end
			isnothing(tel_errors) ? tel_σ = nothing : tel_σ = view(tel_errors, i, :)
			scatter!(plt[2, 1], times_nu, (view(om.tel.lm.s, i, :) .* (norm_M * scaler)) .- (shift_s * (i - 1)); yerror=tel_σ, label=_label, color=plt_colors[c_ind], title="Telluric Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, markerstrokewidth=Int(!isnothing(tel_errors))/2, kwargs...)
			hline!(plt[2, 1], [-(shift_s * (i - 1))]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
	    end
	end
	if plot_stellar
		plot_telluric ? c_offset = inds[end] - 1 : c_offset = 1
		inds = axes(om.star.lm.M, 2)

		# basis plot
		plot_telluric_with_lsf!(plt[1, n_plots], om, om.tel.lm.μ; d=d, color=base_color, alpha=0.3, label=L"\mu_\oplus", title="Stellar Model Feature Vectors", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")
		plot_stellar_with_lsf!(plt[1, n_plots], om, om.star.lm.μ; d=d, color=plt_colors[1], label=L"\mu_\star")

		norm_Ms = [norm(view(om.star.lm.M, :, i)) for i in inds]
		s_std = [std(view(om.star.lm.s, inds[j], :) .* norm_Ms[j]) for j in eachindex(inds)]
		max_s_std = maximum(s_std)
		shift_s = ceil(10 * max_s_std) / 2
		_keys = sort([key for key in keys(df_act)])[1:2:end]
		for i in reverse(eachindex(_keys))
			key = _keys[i]
			y = df_act[key]
			c = max_s_std / std(y)
			scatter!(plt[2, n_plots], times_nu, (c .* (y .- mean(y))) .+ (shift_s*i); label=_keys[i] * " (scaled)", yerror=c.*df_act[key*"_σ"], color=plt_colors[c_ind_f(1 + c_offset)], markerstrokewidth=0.5)
			hline!(plt[2, n_plots], [shift_s*i]; label="", color=plt_colors[c_ind_f(1 + c_offset)], lw=3, alpha=0.4)
			c_offset += 1
		end

		if incl_χ²; holder = copy(om.star.lm.s) end
		# for i in reverse(inds)
		for j in eachindex(inds)
			i = inds[j]
			c_ind = c_ind_f(i + c_offset)
			norm_M = norm_Ms[j]

			# basis plot
			plot_stellar_with_lsf!(plt[1, n_plots], om, (view(om.star.lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); d=d, label="Basis $i", color=plt_colors[c_ind])

			scaler = max_s_std / std(view(om.star.lm.s, i, :) .* norm_M)
			_label = "Weights $i" * scaler_string(scaler)
			# weights plot
			if incl_χ²
				om.star.lm.s[i, :] .= 0
				Δχ² = 1 - (χ²_base / SSOF._loss(o, om, d; star=vec(om.star.lm)))
				om.star.lm.s[i, :] .= view(holder, i, :)
				_label *= " (Δχ² = $(round(Δχ²; digits=3)))"
			end
			isnothing(star_errors) ? star_σ = nothing : star_σ = view(star_errors, i, :)
			scatter!(plt[2, n_plots], times_nu, (view(om.star.lm.s, i, :) .* (norm_M * scaler)) .- (shift_s * (i - 1)); yerror=star_σ, label=_label, color=plt_colors[c_ind], title="Stellar Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, markerstrokewidth=Int(!isnothing(star_errors))/2, kwargs...)
			hline!(plt[2, n_plots], [-shift_s * (i - 1)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
		end
	end

    if display_plt; display(plt) end
    return plt
end
plot_model(mws::SSOF.ModelWorkspace, airmasses::Vector, times_nu::Vector; kwargs...) =
	plot_model(mws.om, airmasses, times_nu; d=mws.d, o=mws.o, kwargs...)
function plot_model(lm::SSOF.FullLinearModel; λ=eachindex(lm.μ), times=axes(lm.s, 2), display_plt::Bool=true, kwargs...)
	plt = _plot(; layout=grid(2, 1), size=(_plt_size[1],_plt_size[2]*1.5), kwargs...)
	shift_M = 0.2
	inds = axes(lm.M, 2)

	# basis plot
	plot!(plt[1, 1], λ, lm.μ; label="μ", title="Model Bases", legend=:outerright, xlabel = "Wavelength (Å)", ylabel = "Continuum Normalized Flux + Const")

	shift_s = ceil(10 * maximum([std(lm.s[inds[i], :] .* norm(lm.M[:, inds[i]])) for i in inds])) / 2
	for i in reverse(inds)
		c_ind = c_ind_f(i - inds[1])
		norm_M = norm(view(lm.M, :, i))

		# basis plot
		plot!(plt[1, 1], λ, (view(lm.M, :, i) ./ norm_M) .- shift_M * (i - 1); label="Basis $i", color=plt_colors[c_ind])

		# weights plot
		_scatter!(plt[2, 1], times, (view(lm.s, i, :) .* norm_M) .- (shift_s * (i - 1)); label="Weights $i", color=plt_colors[c_ind], title="Model Weights", xlabel = "Time (d)", ylabel = "Weights + Const", legend=:outerright, kwargs...)
		hline!(plt[2, 1], [-shift_s * (i - 1)]; label="", color=plt_colors[c_ind], lw=3, alpha=0.4)
	end
	if display_plt; display(plt) end
	return plt
end

time_average(a) = mean(a; dims=2)
function status_plot(mws::SSOF.ModelWorkspace; tracker::Int=0, display_plt::Bool=true, include_χ²::Bool=true, kwargs...)
    o = mws.o
	d = mws.d
	obs_mask = vec(all(.!(isinf.(d.var)); dims=2))
    obs_λ = time_average(exp.(d.log_λ_obs))
    plot_star_λs = time_average(exp.(d.log_λ_star))
	include_χ² ?
		plt = plot_spectrum(; legend = :bottomright, layout = grid(3, 1, heights=[0.6, 0.2, 0.2]), ylabel="Flux + Constant Shift", kwargs...) :
		plt = plot_spectrum(; legend = :bottomright, layout = grid(2, 1, heights=[0.85, 0.15]), kwargs...)


	# TODO: take average after broadening with LSF
	tel_model = time_average(mws.om.tel.lm())
    plot_telluric_with_lsf!(plt[1], mws.om, vec(tel_model); d=mws.d, label="Mean Telluric Model", color=plt_colors[1])
	# tel_model = mws.om.tel.lm()
    # plot_telluric_with_lsf!(plt[1], mws.om, tel_model; d=mws.d, label="Mean Telluric Model", color=plt_colors[1])
    shift = 1.1 - minimum(tel_model)

	star_model = time_average(mws.om.star.lm())
	# star_model = mws.om.star.lm()
	# typeof(mws.om) <: SSOF.OrderModelWobble ?
	# 	star_model = time_average(mws.om.star()) :
	# 	star_model = time_average(mws.om.star() + mws.om.rv())

	plot_stellar_with_lsf!(plt[1], mws.om, vec(star_model .- shift); d=mws.d, label="Mean Stellar Model", color=plt_colors[2])
	# plot_stellar_with_lsf!(plt[1], mws.om, star_model .- shift; d=mws.d, label="Mean Stellar Model", color=plt_colors[2])

    # shift += 1.1 - minimum(star_model)
    # plot!(plt[1], obs_λ, time_average(o.total) .- shift, label="Mean Full Model", color=base_color)

    _scatter!(plt[2], obs_λ[obs_mask], time_average(abs.(view(d.flux, obs_mask, :) - view(o.total, obs_mask, :))), ylabel="MAD", label="", alpha=0.5, color=base_color, xlabel="", ms=1.5)

	if include_χ²
		_scatter!(plt[3], obs_λ, -sum(SSOF._loss_diagnostic(mws); dims=2), ylabel="Remaining χ²", label="", alpha=0.5, color=base_color, xlabel="", ms=1.5)
	end

    if display_plt; display(plt) end
    return plt
end

function component_test_plot(ys::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector; size=(_plt_size[1],_plt_size[2]*1.5), ylabel="ℓ", xguidefontsize = 16, kwargs...)
    plt = _plot(; ylabel=ylabel, layout=grid(2, 1), size=size, xguidefontsize = xguidefontsize, kwargs...)
	# lims = [maximum(ys[.!(isinf.(ys))]), minimum(ys[.!(isinf.(ys))])]
	lims = Array{Float64}(undef, 2)
	lim_inds = ones(Int, 2)

	if ylabel=="ℓ"
		best = argmax(ys)
		if test_n_comp_tel[1:2] == -1:0
			lim_inds[1] = min(3, best[1])
		end
		if test_n_comp_star[1] == 0
			lim_inds[2] = min(2, best[2])
		end
		window = view(ys, lim_inds[1]:length(test_n_comp_tel), lim_inds[2]:length(test_n_comp_star))
		lims[1] = minimum(window[isfinite.(window)])
		lims[2] = ys[best]
	else
		best = argmin(ys)
		if test_n_comp_tel[1:2] == -1:0
			lim_inds[1] = min(3, best[1])
		end
		if test_n_comp_star[1] == 0
			lim_inds[2] = min(2, best[2])
		end
		lims[1] = ys[best]
		window = view(ys, (lim_inds[1]):length(test_n_comp_tel), (lim_inds[2]):length(test_n_comp_star))
		lims[2] = maximum(window[isfinite.(window)])
	end
	buffer = 0.3 * (lims[2] - lims[1])
	ylims!(plt, lims[1] - buffer, lims[2] + buffer)
    for i in eachindex(test_n_comp_tel)
		test_n_comp_tel[i]==-1 ? _label="∅" : _label="$(test_n_comp_tel[i])"
        plot!(plt[1], test_n_comp_star, ys[i, :]; label=_label, leg_title=L"K_\oplus", shape=:diamond, msw=0, xlabel=L"K_\star")
    end
    for i in eachindex(test_n_comp_star)
        plot!(plt[2], test_n_comp_tel, ys[:, i]; label="$(test_n_comp_star[i])", leg_title=L"K_\star", shape=:diamond, msw=0, xlabel=L"K_\oplus")
    end
    # display(plt)
    return plt
end

function save_model_plots(mws, airmasses, times_nu, save_path::String; display_plt::Bool=true, incl_χ²::Bool=true, tel_errors::Union{AbstractMatrix, Nothing}=nothing, star_errors::Union{AbstractMatrix, Nothing}=nothing, df_act::Dict=Dict(), kwargs...)
	plt = plot_model(mws, airmasses, times_nu; display_plt=display_plt, incl_χ²=incl_χ², tel_errors=tel_errors, star_errors=star_errors, df_act=df_act, kwargs...);
	png(plt, save_path * "model.png")

	plt = status_plot(mws; display_plt=display_plt, kwargs...);
	png(plt, save_path * "status_plot.png")
end

function gated_plot!(plt, plotf!::Function, x::AbstractVector, y::AbstractVector, ylims, c, alpha, label, markersize)
	@assert plotf! == scatter! || plotf! == plot!
	m1 = y .< ylims[1]
	scatter!(x[m1], ones(sum(m1)) .* (ylims[1] + .05); label="", c=c, markershape=:utriangle, markerstrokewidth=0, alpha=alpha, markersize=markersize)
	m2 = y .> ylims[2]
	scatter!(x[m2], ones(sum(m2)) .* (ylims[2] - .05); label="", c=c, markershape=:dtriangle, markerstrokewidth=0, alpha=alpha, markersize=markersize)
	m = .!m1 .&& .!m2
	plotf!(x[m], y[m]; label=label, c=c, markerstrokewidth=0, alpha=alpha, markersize=markersize)
end

function data_usage_plot(d::SSOF.Data, bad_inst::Vector, bad_high::Vector, bad_snap::Vector, bad_edge::Vector, bad_isol::Vector, bad_byeye::Vector; save_path::String="", use_var_s::Bool=true)
	if use_var_s
		ever_used = vec(any(isfinite.(d.var_s); dims=2))
		always_used = vec(all(.!(isinf.(d.var_s)); dims=2))
	else
		ever_used = vec(any(isfinite.(d.var); dims=2))
		always_used = vec(all(.!(isinf.(d.var)); dims=2))
	end
	sometimes_used = xor.(ever_used, always_used)
	never_used = .!ever_used
	mean_flux = vec(mean(d.flux; dims=2))
	pixs = axes(d.flux, 1)

	yli = (-.05, 1.5)
	plt = _plot(; title="Data Usage", xlabel="Pixel #", ylabel="Normalized Flux", legend=:outerright, ylims=yli)
	if sum(always_used) > 0; gated_plot!(plt, scatter!, view(pixs, always_used), view(mean_flux, always_used), yli, base_color, 1, "Always used", 1) end

	bads_str = ["Instrumental", "High", "Snappy", "Low SNR", "Isolated", "By Eye"]
	bads = [bad_inst, bad_high, bad_snap, bad_edge, bad_isol, bad_byeye]
	ss = [3, 3, 3, 2, 2, 2]
	for i in eachindex(bads)
		bad = bads[i]
		bad_str = bads_str[i]
		s = ss[i]
		if length(bad) > 0; gated_plot!(plt, scatter!, bad, view(mean_flux, bad), yli, plt_colors[i+1], 0.4, bad_str, s) end
	end

	if sum(sometimes_used) > 0; gated_plot!(plt, scatter!, view(pixs, sometimes_used), view(mean_flux, sometimes_used), yli, plt_colors[1], 0.6, "Sometimes used", 1) end
	if sum(never_used) > 0; gated_plot!(plt, scatter!, view(pixs, never_used), view(mean_flux, never_used), yli, :red, 1, "Never used", 1) end
	if save_path != ""; png(plt, save_path * "data_usage.png") end
	return plt
end
