## Setup
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

## Importing data with Eric's code

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using RvSpectMLBase, RvSpectML
using EchelleInstruments, EchelleInstruments.EXPRES
using CSV, DataFrames, Query, StatsBase, Statistics, Dates

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 2)]
target_subdir = star * "/"   # USER: Replace with directory of your choice
fits_target_str = star
paths_to_search_for_param = ["EXPRES"]
include("data_locs.jl")  # defines expres_data_path and expres_save_path

# NOTE: make_manifest does not update its paths_to_search when default_paths_to_search is defined here, so if you change the line above, you must also include "paths_to_search=default_paths_to_search" in the make_manifest() function call below
pipeline_plan = PipelinePlan()
dont_make_plot!(pipeline_plan, :movie)
reset_all_needs!(pipeline_plan)

target_dir = expres_data_path * target_subdir
target_files = readdir(target_dir)
using FITSIO
n_orders = size(FITS(target_dir * readdir(target_dir)[1])[1])[2]
n_obs = sum(occursin.(r"\.fits$", target_files))
excal_masks = Array{UnitRange, 2}(undef, n_obs, n_orders)
flux_masks = Array{UnitRange, 2}(undef, n_obs, n_orders)

if need_to(pipeline_plan,:read_spectra)
    df_files = make_manifest(expres_data_path, target_subdir, EXPRES)
    # Reading in customized parameters from param.jl.
    eval(code_to_include_param_jl(paths_to_search=paths_to_search_for_param))
    # Reading in FITS files

    all_spectra = Spectra2DExtended[]
	function mask2range(mask::AbstractVector)
		try
			range = findfirst(mask):findlast(mask)
			@assert all(mask[range] .== true)
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
        spectra, excal_mask = EXPRES.read_data(eachrow(df_files_use)[i]; store_min_data=true, store_tellurics=true, normalization=:blaze, return_λ_obs=true, return_excalibur_mask=true)
		append!(all_spectra, [spectra])
		excal_masks[i, :] = mask2range(excal_mask)
		flux_masks[i, :] = mask2range(.!(isnan.(spectra.flux)))
    end
    GC.gc()
    dont_need_to!(pipeline_plan,:read_spectra)
end

flatten_ranges(ranges::AbstractVector) = maximum([range[1] for range in ranges]):minimum([range[end] for range in ranges])
flatten_ranges(ranges::AbstractMatrix) = [flatten_ranges(view(ranges, :, i)) for i in 1:size(ranges, 2)]
excal_inds = flatten_ranges(excal_masks)
flux_inds = flatten_ranges(flux_masks)

times_nu = [s.metadata[:bjd] for s in all_spectra]
airmasses = [parse(Float64, s.metadata[:airmass]) for s in all_spectra]

## Switching to my data format
println("starting to write new files")

# 68 has a bunch of tels, 47 has very few
min_order_width = 1000
for order in 1:n_orders
	save_path = expres_save_path * star * "/$(order)/"
	mkpath(save_path)
	used_excal = length(excal_inds[order]) > min_order_width
	if used_excal
		mask_inds = flatten_ranges([flux_inds[order], excal_inds[order]])
	else
		mask_inds = flux_inds[order]
	end
	if length(mask_inds) > min_order_width
		len_obs = length(mask_inds)
		flux_obs = ones(len_obs, n_obs)
		var_obs = zeros(len_obs, n_obs)
		log_λ_obs = zeros(len_obs, n_obs)
		log_λ_star = zeros(len_obs, n_obs)
		for i in 1:n_obs # 13s
		    flux_obs[:, i] = all_spectra[i].flux[mask_inds, order]
		    var_obs[:, i] = all_spectra[i].var[mask_inds, order]
		    log_λ_obs[:, i] = log.(all_spectra[i].λ_obs[mask_inds, order])
		    log_λ_star[:, i] = log.(all_spectra[i].λ[mask_inds, order])
		end
		data = SSOF.Data(flux_obs, var_obs, log_λ_obs, log_λ_star)
		SSOF.process!(data; order=6)
		# x = copy(data.flux)
		# x[data.var .> 1] .= 0
		# heatmap(x)
		@save save_path*"data.jld2" n_obs data times_nu airmasses used_excal
	else
		println("order $order skipped for being only $(length(mask_inds)) useful pixels wide")
	end
end
