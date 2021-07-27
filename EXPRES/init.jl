## Setup
using Pkg
using JLD2
Pkg.activate("EXPRES")
Pkg.instantiate()

## Importing data with Eric's code

import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using RvSpectMLBase, RvSpectML
using EchelleInstruments, EchelleInstruments.EXPRES
using CSV, DataFrames, Query, StatsBase, Statistics, Dates

stars = ["10700", "26965"]
star = stars[SSOF.parse_args(1, Int, 1)]
target_subdir = star * "/"   # USER: Replace with directory of your choice
fits_target_str = star
paths_to_search_for_param = ["EXPRES"]
include("data_locs.jl")  # defines expres_data_path and expres_save_path

# NOTE: make_manifest does not update its paths_to_search when default_paths_to_search is defined here, so if you change the line above, you must also include "paths_to_search=default_paths_to_search" in the make_manifest() function call below
pipeline_plan = PipelinePlan()
dont_make_plot!(pipeline_plan, :movie)
reset_all_needs!(pipeline_plan)
masks = Array{UnitRange, 2}(undef, sum(occursin.(r"\.fits$", readdir(expres_data_path * target_subdir))), 86)

if need_to(pipeline_plan,:read_spectra)
    df_files = make_manifest(expres_data_path, target_subdir, EXPRES)
    # Reading in customized parameters from param.jl.
    eval(code_to_include_param_jl(paths_to_search=paths_to_search_for_param))
    # Reading in FITS files
    all_spectra = Spectra2DExtended[]
    for j in 1:size(masks, 1)
        row = eachrow(df_files_use)[j]
        spectra, mask = EXPRES.read_data(row; store_min_data=true, store_tellurics=true, normalization=:blaze, return_λ_obs=true, return_excalibur_mask=true)
        append!(all_spectra, [spectra])
        for i in 1:size(mask, 2)
            try
                x = findfirst(view(mask, :, i)):findlast(view(mask, :, i))
                masks[j, i] = x
                @assert all(mask[x, i] .== true)
            catch y
                if typeof(y)==MethodError
                    masks[j, i] = 0:0
                else
                    throw(y)
                end
            end
        end
    end
    GC.gc()
    dont_need_to!(pipeline_plan,:read_spectra)
end

inds = Vector{UnitRange}(undef, size(masks,2))
for i in 1:length(inds)
    inds[i] = maximum([mask[1] for mask in masks[:, i]]):minimum([mask[end] for mask in masks[:, i]])
end

times_nu = [s.metadata[:bjd] for s in all_spectra]
airmasses = [parse(Float64, s.metadata[:airmass]) for s in all_spectra]

## Switching to my data format
println("starting to write new files")

useful_orders = [length(i) for i in inds] .> 1000

# 68 has a bunch of tels, 47 has very few
n_obs = length(all_spectra)
for desired_order in findfirst(useful_orders):findlast(useful_orders)
	save_path = expres_save_path * star * "/$(desired_order)/"
	mkpath(save_path)
	mask_inds = inds[desired_order]

	len_obs = length(mask_inds)
	flux_obs = ones(len_obs, n_obs)
	var_obs = zeros(len_obs, n_obs)
	log_λ_obs = zeros(len_obs, n_obs)
	log_λ_star = zeros(len_obs, n_obs)
	for i in 1:n_obs # 13s
	    flux_obs[:, i] = all_spectra[i].flux[mask_inds, desired_order]
	    var_obs[:, i] = all_spectra[i].var[mask_inds, desired_order]
	    log_λ_obs[:, i] = log.(all_spectra[i].λ_obs[mask_inds, desired_order])
	    log_λ_star[:, i] = log.(all_spectra[i].λ[mask_inds, desired_order])
	end
	data = SSOF.Data(flux_obs, var_obs, log_λ_obs, log_λ_star)
	SSOF.process!(data; order=6)
	@save save_path*"data.jld2" n_obs data times_nu airmasses
end

# lfc_orders = 43:72
# order_list_timeseries = extract_orders(all_spectra, pipeline_plan; orders_to_use=lfc_orders, recalc=true)
