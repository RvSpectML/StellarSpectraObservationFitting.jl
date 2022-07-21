import StellarSpectraObservationFitting as SSOF
using JLD2
using CSV, DataFrames, Query, StatsBase, Statistics, Dates
using FITSIO
using Statistics
using Plots


function neid_extras(df_files::DataFrame, save_path_base::String)
	n_obs = nrow(df_files)
	f = FITS(df_files.Filename[1])
	neid_tel = zeros(n_obs, size(f[11], 1), size(f[11], 2))
	neid_tel_wvapor = zeros(n_obs)
	neid_tel_zenith = zeros(n_obs)
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
		_df_h = read_header(f[11])
		neid_tel_wvapor[i] = _df_h["WVAPOR"]
		neid_tel_zenith[i] = _df_h["ZENITH"]
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
	@save save_path_base * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv d_act_tot neid_tel d_lcs neid_tel_wvapor neid_tel_zenith
end

function neid_activity_indicators(pipeline_path::String, data::SSOF.Data)
	@load pipeline_path d_act_tot d_lcs
	lo, hi = exp.(quantile(vec(data.log_λ_star), [0.05, 0.95]))
	df_act = Dict()
	for wv in keys(d_lcs)
		if lo < parse(Float64, wv) < hi
			for key in d_lcs[wv]
				df_act[key] = d_act_tot[key]
				df_act[key*"_σ"] = d_act_tot[key*"_σ"]
			end
		end
	end
	return df_act
end

function neid_plots(mws::SSOF.ModelWorkspace,
	airmasses::AbstractVector,
	times_nu::AbstractVector,
	rvs::AbstractVector,
	rv_errors::AbstractVector,
	star::String,
	base_path::String,
	pipeline_path::String,
	desired_order::Int;
	mask = :,
	display_plt::Bool=false,
	tel_errors::Union{AbstractMatrix, Nothing}=nothing,
	star_errors::Union{AbstractMatrix, Nothing}=nothing,
	df_act::Dict=Dict())

	@load pipeline_path neid_time neid_rv neid_rv_σ neid_order_rv
	neid_time .-= 2400000.5

	# Compare RV differences to actual RVs from activity
	plt = plot_model_rvs(view(times_nu, mask), view(rvs, mask), view(rv_errors, mask), view(neid_time, mask), view(neid_rv, mask), view(neid_rv_σ, mask); display_plt=display_plt, title="$star (median σ: $(round(median(vec(view(rv_errors, mask))), digits=3)))");
	png(plt, base_path * "model_rvs.png")

	save_model_plots(mws, airmasses, times_nu, base_path; display_plt=display_plt, tel_errors=tel_errors, star_errors=star_errors, df_act=df_act);

	if all(.!iszero.(view(neid_order_rv, :, desired_order)))
	    plt = plot_model_rvs(view(times_nu, mask), view(rvs, mask), view(rv_errors, mask), view(neid_time, mask), view(neid_order_rv, mask, desired_order), zeros(length(view(neid_time, mask))); display_plt=display_plt, title="$star (median σ: $(round(median(vec(rv_errors)), digits=3)))");
	    png(plt, base_path * "model_rvs_order.png")
	end
end
