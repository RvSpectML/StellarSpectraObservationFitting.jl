## Importing data with Eric's code

import StellarSpectraObservationFitting as SSOF
using JLD2
using RvSpectMLBase, RvSpectML
using EchelleInstruments, EchelleInstruments.NEID, EchelleInstruments.EXPRES
using CSV, DataFrames, Query, StatsBase, Statistics, Dates
using FITSIO

function mask2range(mask::AbstractVector)
	try
		range = findfirst(mask):findlast(mask)
		# @assert all(mask[range] .== true)
		return range
	catch exception
		if typeof(exception)==MethodError
			return 0:0
		else
			throw(exception)
		end
	end
end
mask2range(masks::AbstractMatrix) = [mask2range(view(masks, :, i)) for i in 1:size(masks, 2)]

function reformat_spectra(
		df_files::DataFrame,
		save_path_base::String,
		RVSMLInstrument::Module,
		orders_to_read::UnitRange,
		star::String;
		lsf_f::Union{Function,Nothing}=nothing,
		interactive::Bool=false,
		kwargs...)

	# deals with some instrument specific features
	is_EXPRES = RVSMLInstrument == EXPRES
	is_NEID = RVSMLInstrument == NEID

	# Finding data files

	n_obs = nrow(df_files)
	first_order = orders_to_read[1]
	n_orders = length(orders_to_read)

	# intializing masks for bad edges of orders
	if is_EXPRES; excal_masks = Array{UnitRange, 2}(undef, n_obs, n_orders) end
	flux_masks = Array{UnitRange, 2}(undef, n_obs, n_orders)

	all_spectra = Spectra2DExtended[]
	times_to_use = 1:n_obs

	for i in times_to_use # at every time
		if is_EXPRES
			spectra, excal_mask = RVSMLInstrument.read_data(eachrow(df_files)[i]; store_min_data=true, store_tellurics=true, normalization=:blaze, return_λ_obs=true, return_excalibur_mask=true)
			excal_masks[i, :] .= mask2range(excal_mask)
		else
			spectra = RVSMLInstrument.read_data(eachrow(df_files)[i], orders_to_read; normalization=:blaze, return_λ_obs=true)
		end
		append!(all_spectra, [spectra])
		flux_masks[i, :] .= mask2range(.!(isnan.(spectra.flux)))
	end

	if is_EXPRES; excal_inds = SSOF.flatten_ranges(excal_masks) end
	flux_inds = SSOF.flatten_ranges(flux_masks)

	times_nu = [s.metadata[:bjd] for s in all_spectra]
	airmasses = [s.metadata[:airmass] for s in all_spectra]
	# if is_EXPRES; airmasses = [parse(Float64, s.metadata[:airmass]) for s in all_spectra] end

	## Switching to my data format
	println("starting to write new files")

	min_order_width = 1000
	masks_inds = copy(flux_inds)
	order_inds = Int64[]
	for order_ind in 1:n_orders
		if is_EXPRES && length(excal_inds[order_ind]) > min_order_width
			masks_inds[order_ind] = SSOF.flatten_ranges([flux_inds[order_ind], excal_inds[order_ind]])
		end
		if length(masks_inds[order_ind]) > min_order_width
			append!(order_inds, [order_ind])
		else
			println("order $order skipped for being only $(length(mask_inds)) (<$min_order_width) useful pixels wide")
		end
	end

	## flagging weird drift points
	first_λs = Array{Float64}(undef, length(order_inds), n_obs)
	first_masks_inds = [mask_inds[1] for mask_inds in masks_inds]
	for i in times_to_use # 13s
		first_λs[:, i] = log.(all_spectra[i].λ_obs[first_masks_inds, :][order_inds])
	end
	mean_first_λs = vec(mean(first_λs; dims=1))
	drift_mask, drift_stats = SSOF.outlier_mask(mean_first_λs; prop=4.5/n_obs, thres=5, return_stats=true)
	worst_points = sortperm(drift_stats; rev=true)
	if interactive
		annot = text.(times_to_use, :center, :black, 6)
		plt = _plot(; title="Average First log(λ)", xlabel="Time", ylabel="log(λ)", legend=:outerright)
		scatter!(plt, times_nu[drift_mask], mean_first_λs[drift_mask]; label="Good data", series_annotations=annot[drift_mask], markerstrokewidth=0)
		scatter!(plt, times_nu[.!drift_mask], mean_first_λs[.!drift_mask]; label="Potential data to remove", series_annotations=annot[.!drift_mask], markerstrokewidth=0)
		display(plt)
		println("How many points have bad drifts and should be removed?")
		println("Points sorted by anomalousness")
		println(worst_points, "\n")
		answer = readline()
		n_points_remove = parse(Int, answer)
		if n_points_remove > 0
			points_to_remove = worst_points[1:n_points_remove]
			times_to_use = [i for i in times_to_use if !(i in points_to_remove)]
			for i in points_to_remove
				println("ignoring $(df_files.Filename[i])")
			end
		end
	else
		for i in 1:n_obs
			if !drift_mask[i]
				println("$(df_files.Filename[i]) has a weird drift, consider removing it from your analysis")
			end
		end
	end

	## flagging low snr points
	if is_NEID
		snr = [read_header(FITS(fn)[1])["EXTSNR"] for fn in df_files.Filename]
	else
		@error "TODO add SNR for other intruments"
	end
	snr_mask, snr_stats = SSOF.outlier_mask(snr; prop=4.5/n_obs, thres=5, return_stats=true, only_low=true)
	worst_points = sortperm(snr_stats)
	if interactive
		annot = text.(1:n_obs, :center, :black, 6)
		plt = _plot(; title="SNR", xlabel="Time", ylabel="SNR", legend=:outerright)
		scatter!(plt, times_nu[snr_mask], snr[snr_mask]; label="Good data", series_annotations=annot[snr_mask], markerstrokewidth=0)
		scatter!(plt, times_nu[.!snr_mask], snr[.!snr_mask]; label="Potential data to remove", series_annotations=annot[.!snr_mask], markerstrokewidth=0)
		display(plt)
		println("How many low snr points should be removed?")
		println("Points sorted by snr")
		println(worst_points, "\n")
		answer = readline()
		n_points_remove = parse(Int, answer)
		if n_points_remove > 0
			points_to_remove = worst_points[1:n_points_remove]
			times_to_use = [i for i in times_to_use if !(i in points_to_remove)]
			for i in points_to_remove
				println("ignoring $(df_files.Filename[i])")
			end
		end
	else
		for i in 1:n_obs
			if !snr_mask[i]
				println("$(df_files.Filename[i]) has a low snr, consider removing it from your analysis")
			end
		end
	end
	n_obs = length(times_to_use)
	times_nu = times_nu[times_to_use]
	airmasses = airmasses[times_to_use]
	for order_ind in order_inds
		mask_inds = masks_inds[order_ind]
		order = order_ind + first_order - 1
		save_path = save_path_base * "/$(order)/"
		mkpath(save_path)
		len_obs = length(mask_inds)
		flux_obs = ones(len_obs, n_obs)
		var_obs = Array{Float64}(undef, len_obs, n_obs)
		log_λ_obs = Array{Float64}(undef, len_obs, n_obs)
		log_λ_star = Array{Float64}(undef, len_obs, n_obs)
		for i in 1:length(times_to_use) # 13s
			j = times_to_use[i]
			flux_obs[:, i] .= all_spectra[j].flux[mask_inds, order_ind]
			var_obs[:, i] .= all_spectra[j].var[mask_inds, order_ind]
			log_λ_obs[:, i] .= log.(all_spectra[j].λ_obs[mask_inds, order_ind])
			log_λ_star[:, i] .= log.(all_spectra[j].λ[mask_inds, order_ind])
		end
		nan_mask = isnan.(var_obs)
		@assert all(nan_mask .== isnan.(flux_obs))
		flux_obs[nan_mask] .= 1
		var_obs[nan_mask] .= Inf
		if lsf_f != nothing
			is_NEID ?
				data = SSOF.LSFData(flux_obs, var_obs, log_λ_obs, log_λ_star, lsf_f(order)) :
				data = SSOF.LSFData(flux_obs, var_obs, log_λ_obs, log_λ_star, lsf_f(exp.(log_λ_obs), order))
		else
			data = SSOF.GenericData(flux_obs, var_obs, log_λ_obs, log_λ_star)
		end
		if is_NEID; neid_order_masks!(data, order, star) end
		# data_backup = copy(data)

		SSOF.process!(data; order=2, kwargs...)
		@save save_path*"data.jld2" n_obs data times_nu airmasses
		data_usage_plot(data; save_path=save_path)
		# plt = _plot(;size=(2 * _plt_size[1],_plt_size[2]), legend=:bottom)
		# for j in 1:size(data_backup.flux, 2)
		# 	ys = data_backup.flux[:, j]
		# 	nanmask = .!(isnan.(ys))
		# 	ys ./= median(ys[nanmask])
		# 	plot!(plt, xs, ys ./ data.flux[:, j]; label="")
		# end
		println("finished order $order")
	end
end
