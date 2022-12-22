import StellarSpectraObservationFitting as SSOF
using JLD2
using CSV, DataFrames, Query, StatsBase, Statistics, Dates
using FITSIO
using Statistics
using Plots

function n2s(i)
	@assert -1 < i < 1000
	ans = string(i)
	return "0"^(3-length(ans))*ans
end

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
	for i in eachindex(lcs)
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
	neid_time = zeros(n_obs)
	neid_rv = zeros(n_obs)
	neid_rv_σ = zeros(n_obs)
	neid_order_rv = zeros(n_obs, 118)
	for i in 1:n_obs # at every time
		f = FITS(df_files.Filename[i])
		driftfun = read_header(f[1])["DRIFTFUN"]
		if driftfun != "dailymodel0"
			println("spectrum $i ($(df_files.Filename[i])) has a wavelength calib. drift func \"$driftfun\" instead of \"dailymodel0\", consider removing it from your analysis")
		end
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
		ccfs_exist = vec(.!all(iszero.(read(f[13])); dims=1))
		# neid_rv_ords = [j for j in eachindex(ccfs_exist) if ccfs_exist[j]]
		# neid_rv_ords = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 91, 92, 93, 95, 96]
		# println(ccf_header)
		for j in eachindex(ccfs_exist)
		    if ccfs_exist[j]
				# neid_order_rv[i, j] = ccf_header["CCFRV"*n2s(j+51)] * 1000  # m/s
				neid_order_rv[i, j] = ccf_header["CCFRV"*n2s(174-j)] * 1000  # m/s
		    end
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
	base_path::String,
	pipeline_path::String,
	desired_order::Int;
	mask = :,
	display_plt::Bool=false,
	tel_errors::Union{AbstractMatrix, Nothing}=nothing,
	star_errors::Union{AbstractMatrix, Nothing}=nothing,
	df_act::Dict=Dict(),
	title="",
	kwargs...)

	@load pipeline_path neid_time neid_rv neid_rv_σ neid_order_rv
	neid_time .-= 2400000.5

	# Compare RV differences to actual RVs from activity
	plt = plot_model_rvs(view(times_nu, mask), view(rvs, mask), view(rv_errors, mask), view(neid_time, mask), view(neid_rv, mask), view(neid_rv_σ, mask); display_plt=display_plt, title=title, inst_str="NEID");
	png(plt, base_path * "model_rvs.png")

	save_model_plots(mws, airmasses, times_nu, base_path; display_plt=display_plt, tel_errors=tel_errors, star_errors=star_errors, df_act=df_act, kwargs...);

	if all(.!iszero.(view(neid_order_rv, :, desired_order)))
	    plt = plot_model_rvs(view(times_nu, mask), view(rvs, mask), view(rv_errors, mask), view(neid_time, mask), view(neid_order_rv, mask, desired_order), zeros(length(view(neid_time, mask))); display_plt=display_plt, title=title, inst_str="NEID (single order)");
	    png(plt, base_path * "model_rvs_order.png")
	end
end


# used to easily identify how far to mask https://apps.automeris.io/wpd/
function neid_order_masks!(data::SSOF.Data, order::Int, star::String)
	if star=="26965"
		if order==31
		    return SSOF.mask_stellar_feature!(data, log(4326.7), 100)
		elseif order==38
		    return SSOF.mask_stellar_feature!(data, log(4549.5), 100)
		elseif order==41
		    return SSOF.mask_stellar_feature!(data, log(4651.9), 100)
		elseif order==47
		    return SSOF.mask_stellar_feature!(data, log(4871.6), 100)
		elseif order==48
		    return SSOF.mask_stellar_feature!(data, log(4909.8), 100)
		elseif order==60
		    return SSOF.mask_stellar_feature!(data, 0, log(5325.7))
		elseif order==61
			# not sure which it is
		    return SSOF.mask_stellar_feature!(data, 0, log(5374))
			# affected = SSOF.mask_telluric_feature!(data, 0, log(5374.5))
		elseif order==95
		    return SSOF.mask_stellar_feature!(data, log(7832.6), 100)
		end
	end
	return Int[]
end

# neid_lsf_orders = 54:112
neid_temporal_gp_lsf_λs = [190041.16749513964, 190209.46819591062, 190372.75573431773, 190531.030110361, 190684.29132404033, 190832.53937535582, 190975.77426430746, 191113.9959908952, 191247.2045551191, 191375.39995697912, 191498.58219647527, 191616.75127360752, 191729.90718837592, 191838.04994078045, 191941.1795308211, 192039.29595849788, 192132.3992238108, 192220.48932675982, 192303.566267345, 192381.6300455663, 192454.68066142374, 192522.71811491728, 192585.74240604695, 192643.75353481277, 192696.7515012147, 192744.7363052528, 192787.70794692697, 192825.6664262373, 192858.61174318375, 192886.54389776633, 192909.46288998506, 192927.36871983987, 192940.26138733086, 192948.14089245794, 192951.00723522116, 193231.27351045443, 193305.43446253656, 193316.37130324845, 193264.08403259012, 193148.5726505615, 192969.8371571626, 192727.87755239347, 192422.69383625407, 192054.28600874444, 191622.6540698645, 191127.79801961436, 190569.71785799394, 189948.41358500326, 189263.8852006423, 188516.1327049111, 187705.15609780964, 186830.955379338, 185893.530549496, 184892.8816082838, 183829.0085557013, 182701.91139174852, 181511.5901164256, 180258.04472973233, 178941.27523166878]
neid_temporal_gp_lsf_λs_max = maximum(neid_temporal_gp_lsf_λs)

function neid_neid_temporal_gp_lsf_λ(order::Int; fudge_factor::Real=1.01)
	if 53 < order < 113
		return neid_temporal_gp_lsf_λs[order - 53] * fudge_factor
	else
		return neid_temporal_gp_lsf_λs_max * fudge_factor
	end
end


174-7
174-118

length(7:118)

# # expres_lsf_orders = 37:75
# expres_temporal_gp_lsf_λs = [182676.53481102452, 182374.65263197446, 182092.35241142692, 181829.63414938204, 181586.49784583968, 181362.94350079997, 181158.9711142628, 180974.58068622823, 180809.77221669623, 180664.54570566685, 180538.90115314, 180432.8385591158, 180346.35792359413, 180279.45924657505, 180232.1425280586, 180204.4077680447, 180196.2549665334, 180207.68412352467, 180238.69523901856, 180289.288313015, 180359.46334551406, 180449.22033651566, 180558.55928601988, 180687.48019402666, 180835.98306053603, 181004.06788554802, 181191.73466906254, 181398.9834110797, 181625.8141115994, 181872.22677062172, 182138.2213881466, 182423.7979641741, 182728.95649870415, 183053.69699173683, 183398.01944327206, 183761.92385330988, 184145.41022185027, 184548.47854889327, 184971.12883443886]
# expres_temporal_gp_lsf_λs_max = maximum(expres_temporal_gp_lsf_λs)
#
# function expres_neid_temporal_gp_lsf_λ(order::Int; fudge_factor::Real=1.01)
# 	if 36 < order < 76
# 		return expres_temporal_gp_lsf_λs[order - 36] * fudge_factor
# 	else
# 		return expres_temporal_gp_lsf_λs_max * fudge_factor
# 	end
# end
