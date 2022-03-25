## Setup
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

## Importing data with Eric's code

using JLD2
using RvSpectMLBase, RvSpectML
import StellarSpectraObservationFitting as SSOF
using EchelleInstruments, EchelleInstruments.NEID
using CSV, DataFrames, Query, StatsBase, Statistics, Dates

stars = ["10700"]
star = stars[SSOF.parse_args(1, Int, 1)]
target_subdir = star * "/"   # USER: Replace with directory of your choice
fits_target_str = "HD " * star  # needed by param.jl
paths_to_search_for_param = ["NEID"]
include("data_locs.jl")  # defines expres_data_path and expres_save_path
# include("lsf.jl")  # defines EXPRES_lsf()

# NOTE: make_manifest does not update its paths_to_search when default_paths_to_search is defined here, so if you change the line above, you must also include "paths_to_search=default_paths_to_search" in the make_manifest() function call below
pipeline_plan = PipelinePlan()
dont_make_plot!(pipeline_plan, :movie)
reset_all_needs!(pipeline_plan)

target_dir = neid_data_path * target_subdir
target_files = readdir(target_dir)
using FITSIO
first_order = min_order(NEID2D())
orders_to_read = first_order:max_order(NEID2D())
n_orders = length(orders_to_read)
n_obs = sum(occursin.(r"\.fits$", target_files))
# excal_masks = Array{UnitRange, 2}(undef, n_obs, n_orders)
flux_masks = Array{UnitRange, 2}(undef, n_obs, n_orders)

if need_to(pipeline_plan,:read_spectra)
	df_files = make_manifest(neid_data_path, target_subdir, NEID)
	# Reading in customized parameters from param.jl.
	eval(code_to_include_param_jl(paths_to_search=paths_to_search_for_param))
	# Reading in FITS files

	all_spectra = Spectra2DExtended[]
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

	for i in 1:n_obs # at every time
		# spectra, excal_mask = NEID.read_data(eachrow(df_files_use)[i]; store_min_data=true, store_tellurics=true, normalization=:blaze, return_λ_obs=true, return_excalibur_mask=true)
		spectra = NEID.read_data(eachrow(df_files_use)[i], orders_to_read; normalization=:blaze, return_λ_obs=true)
		append!(all_spectra, [spectra])
		flux_masks[i, :] .= mask2range(.!(isnan.(spectra.flux)))
	end
	GC.gc()
	dont_need_to!(pipeline_plan,:read_spectra)
end

flatten_ranges(ranges::AbstractVector) = maximum([range[1] for range in ranges]):minimum([range[end] for range in ranges])
flatten_ranges(ranges::AbstractMatrix) = [flatten_ranges(view(ranges, :, i)) for i in 1:size(ranges, 2)]
flux_inds = flatten_ranges(flux_masks)

times_nu = [s.metadata[:bjd] for s in all_spectra]
airmasses = [s.metadata[:airmass] for s in all_spectra]

## Switching to my data format
println("starting to write new files")

min_order_width = 1000
for order_ind in 1:n_orders
	order = order_ind + first_order - 1
	save_path = neid_save_path * star * "/$(order)/"
	mkpath(save_path)
	mask_inds = flux_inds[order_ind]

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
		data = SSOF.GenericData(flux_obs, var_obs, log_λ_obs, log_λ_star)
		SSOF.process!(data; order=6)
		# x = copy(data.flux)
		# x[data.var .> 1] .= 0
		# heatmap(x)
		@save save_path*"data.jld2" n_obs data times_nu airmasses
	else
		println("order $order skipped for being only $(length(mask_inds)) useful pixels wide")
	end
	println("finished order $order")
end


# saving NEID pipeline results
function n2s(i)
	@assert -1 < i < 1000
	ans = string(i)
	return "0"^(3-length(ans))*ans
end
rv_ords = 57:122

neid_time = zeros(n_obs)
neid_rv = zeros(n_obs)
neid_rv_σ = zeros(n_obs)
neid_order_rv = zeros(n_obs, orders_to_read[end])
for i in 1:n_obs # at every time
	ccf_header = read_header(FITS(df_files_use.Filename[i])[13])
	neid_time[i] = ccf_header["CCFJDMOD"]
	neid_rv[i] = ccf_header["CCFRVMOD"] * 1000  # m/s
	neid_rv_σ[i] = ccf_header["DVRMSMOD"] * 1000  # m/s
	for j in rv_ords
		neid_order_rv[i, j] = ccf_header["CCFRV"*n2s(j)] * 1000  # m/s
	end
end
ord_has_rvs = vec(all(.!iszero.(neid_order_rv); dims=2))
@save neid_save_path * star * "/neid_pipeline.jld2" neid_time neid_rv neid_rv_σ neid_order_rv ord_has_rvs
