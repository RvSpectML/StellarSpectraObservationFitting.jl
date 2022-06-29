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

function n2s(i)
	@assert -1 < i < 1000
	ans = string(i)
	return "0"^(3-length(ans))*ans
end

function reformat_spectra(
		df_files::DataFrame,
		save_path_base::String,
		RVSMLInstrument::Module,
		orders_to_read::UnitRange;
		lsf_f::Union{Function,Nothing}=nothing)

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

	for i in 1:n_obs # at every time
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
	is_EXPRES ?
		airmasses = [parse(Float64, s.metadata[:airmass]) for s in all_spectra] :
		airmasses = [s.metadata[:airmass] for s in all_spectra]

	## Switching to my data format
	println("starting to write new files")

	min_order_width = 1000
	for order_ind in 1:n_orders
		order = order_ind + first_order - 1
		save_path = save_path_base * "/$(order)/"
		mkpath(save_path)
		is_EXPRES ?
			used_excal = length(excal_inds[order_ind]) > min_order_width :
			used_excal = false
		if used_excal
			mask_inds = SSOF.flatten_ranges([flux_inds[order_ind], excal_inds[order_ind]])
		else
			mask_inds = flux_inds[order_ind]
		end

		if length(mask_inds) > min_order_width
			len_obs = length(mask_inds)
			flux_obs = ones(len_obs, n_obs)
			var_obs = Array{Float64}(undef, len_obs, n_obs)
			log_λ_obs = Array{Float64}(undef, len_obs, n_obs)
			log_λ_star = Array{Float64}(undef, len_obs, n_obs)
			for i in 1:n_obs # 13s
				flux_obs[:, i] .= all_spectra[i].flux[mask_inds, order_ind]
				var_obs[:, i] .= all_spectra[i].var[mask_inds, order_ind]
				log_λ_obs[:, i] .= log.(all_spectra[i].λ_obs[mask_inds, order_ind])
				log_λ_star[:, i] .= log.(all_spectra[i].λ[mask_inds, order_ind])
			end
			lsf_f != nothing ?
				data = SSOF.LSFData(flux_obs, var_obs, log_λ_obs, log_λ_star, lsf_f(exp.(log_λ_obs))) :
				data = SSOF.GenericData(flux_obs, var_obs, log_λ_obs, log_λ_star)
			# data_backup = copy(data)

			SSOF.process!(data; order=2)
			@save save_path*"data.jld2" n_obs data times_nu airmasses
			# plt = _plot(;size=(2 * _plt_size[1],_plt_size[2]), legend=:bottom)
			# for j in 1:size(data_backup.flux, 2)
			# 	ys = data_backup.flux[:, j]
			# 	nanmask = .!(isnan.(ys))
			# 	ys ./= median(ys[nanmask])
			# 	plot!(plt, xs, ys ./ data.flux[:, j]; label="")
			# end
			println("finished order $order")
		else
			println("order $order skipped for being only $(length(mask_inds)) useful pixels wide")
		end
	end

	if is_NEID; neid_extras(df_files, save_path_base) end
end

function neid_extras(df_files::DataFrame, save_path_base::String)
	n_obs = nrow(df_files)
	f = FITS(df_files.Filename[1])
	neid_tel = zeros(n_obs, size(f[11], 1), size(f[11], 2))
	_df = DataFrame(f[14])
	inds = _df.INDEX
	lcs = _df.LINE_CENTER
	d_lcs = Dict()
	for i in 1:length(lcs)
		_lcs = split(lcs[i], ", ")
		for j in _lcs
			if haskey(d_lcs, j)
				append!(d_lcs[j], [inds[i]])
			else
				d_lcs[j] = [inds[i]]
			end
		end
	end
	Hα = "6562.808"
	if Hα in keys(d_lcs) && "Ha06_2" in d_lcs[Hα]; d_lcs[Hα] = ["Ha06_2"] end
	df_cols = String[]
	for i in inds
		append!(df_cols, [i, i*"_σ"])
	end
	df_act = zeros(n_obs, length(df_cols))
	rv_ords = 57:122
	neid_time = zeros(n_obs)
	neid_rv = zeros(n_obs)
	neid_rv_σ = zeros(n_obs)
	neid_order_rv = zeros(n_obs, 122)
	for i in 1:n_obs # at every time
		f = FITS(df_files.Filename[i])
		neid_tel[i, :, :] .= read(f[11])
		_df = DataFrame(f[14])
		df_act[i, 1:2:end] = _df.VALUE
		df_act[i, 2:2:end] = _df.UNCERTAINTY
		ccf_header = read_header(f[13])
		neid_time[i] = ccf_header["CCFJDMOD"]
		neid_rv[i] = ccf_header["CCFRVMOD"] * 1000  # m/s
		neid_rv_σ[i] = ccf_header["DVRMSMOD"] * 1000  # m/s
		for j in rv_ords
			neid_order_rv[i, j] = ccf_header["CCFRV"*n2s(j)] * 1000  # m/s
		end
	end
	d_act_tot = Dict()
	for i in 1:2:length(df_cols)
		d_act_tot[df_cols[i]] = df_act[:, i]
		d_act_tot[df_cols[i+1]] = df_act[:, i+1]
	end
	@save save_path_base * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv d_act_tot neid_tel d_lcs
end
