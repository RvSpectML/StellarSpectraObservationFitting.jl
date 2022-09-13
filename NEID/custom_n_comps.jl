# using Pkg
# Pkg.activate("NEID")
# # Pkg.instantiate()
#
# import StellarSpectraObservationFitting as SSOF
# using DataFrames, CSV
# using JLD2
#
# stars = ["10700", "26965", "22049", "3651", "2021/12/19", "2021/12/20", "2021/12/23"]
# star_choice = SSOF.parse_args(1, Int, 2)
# star = stars[star_choice]
# include("NEID/data_locs.jl")  # defines neid_data_path and neid_save_path
# df = DataFrame(
# 	order=Int[],
# 	n_tel_aic=Int[],
# 	n_star_aic=Int[],
# 	better_model=Int[],
# 	n_tel_by_eye=Int[],
# 	n_star_by_eye=Int[],
# 	has_reciprocal_continuum=Bool[])
#
# orders = 4:118
# for i in 1:length(orders)
# 	order=orders[i]
# 	base_path = neid_save_path * star * "/$(order)/log_vil_lsf/"
# 	save_fn = base_path * "results.jld2"
# 	decision_fn = base_path * "model_decision.jld2"
#  	try
#         @load save_fn model
#         @load decision_fn best_ind better_models
#         better_model = better_models[best_ind]
#         if all(isone.(model.tel.lm.μ)) && !SSOF.is_time_variable(model.tel)  # TODO check this
#             n_tel = -1
#         elseif SSOF.is_time_variable(model.tel)
#             n_tel = size(model.tel.lm.M, 2)
#         else
#             n_tel = 0
#         end
#         if SSOF.is_time_variable(model.star)
#             n_star = size(model.star.lm.M, 2)
#         else
#             n_star = 0
#         end
# 		push!(df, (order, n_tel, n_star, better_model, n_tel, n_star, false))
#     catch err
#         if isa(err, SystemError)
#             println(save_fn * " is missing")
# 			push!(df, (order, -3, -3, 1, -3, -3, false))
#         elseif isa(err, KeyError)
#             println(save_fn * " is incomplete")
# 			push!(df, (order, -2, -2, 1, -2, -2, false))
#         else
#             rethrow()
#         end
#     end
# end
# CSV.write("n_comps.csv", df)

using Pkg
Pkg.activate("NEID")

using DataFrames, CSV

stars = ["10700", "26965", "22049", "3651", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
include("data_locs.jl")  # defines neid_data_path and neid_save_path
df = DataFrame(CSV.File("NEID/n_comps.csv"))
if "redo" in names(df)
	df[:, :redo] .= false
else
	df.redo .= false
end
function n_comp_by_eye!(df::DataFrame, order::Int; n_tel=nothing, n_star=nothing, has_reciprocal_continuum=nothing)
	i = order - 3
	@assert df[i, :order] == order
	if !isnothing(n_tel); df[i, :n_tel_by_eye] = n_tel end
	if !isnothing(n_star); df[i, :n_star_by_eye] = n_star end
	if !isnothing(has_reciprocal_continuum); df[i, :has_reciprocal_continuum] = has_reciprocal_continuum end
	df[i, :redo] = true
end

# wacky basis vectors
n_comp_by_eye!(df, 10; n_star=0)  # weird downward slope thing halfway through
n_comp_by_eye!(df, 13; n_star=0)  # weird downward slope thing halfway through
n_comp_by_eye!(df, 18; n_star=1)  # weird downward slope thing halfway through?
# n_comp_by_eye!(df, 27; n_star=0)  # noisy basis, doesn't seem to affect RVs
n_comp_by_eye!(df, 33; n_star=0)  # noisy basis, does seem to affect RVs
n_comp_by_eye!(df, 44; n_star=0)  # wiggly basis, does seem to slightly affect RVs
n_comp_by_eye!(df, 45; n_star=0)  # wiggly basis, does seem to slightly affect RVs (but worse)
n_comp_by_eye!(df, 56; n_tel=0, n_star=0)  # just utterly confused but don't expect much improvement
n_comp_by_eye!(df, 57; n_star=0)  # two weird dopplery basis vectors
# n_comp_by_eye!(df, 76; n_tel=?, n_star=?)  # has o2 and h2o and barycenter basis vector, rv or aic dont point in clear direction, great illustrative order
# n_comp_by_eye!(df, 77; n_star=0, has_reciprocal_continuum=true)  # has o2 and h2o and barycenter basis vector, could have 0 or 1 bases
n_comp_by_eye!(df, 83; n_tel=1, n_star=0)  # doppler basis vector? 1 tel aic preffered if no stellar bases, could also do n_tel=-1
n_comp_by_eye!(df, 84; n_star=0, has_reciprocal_continuum=true)  # barycenter and continuum bases vector? rv clearly prefers no stellar
n_comp_by_eye!(df, 85; n_tel=2, n_star=0, has_reciprocal_continuum=true)  # 4 stellar bases, it got confused
n_comp_by_eye!(df, 86; n_star=0, has_reciprocal_continuum=true)  # continuum basis vector
n_comp_by_eye!(df, 88; n_tel=2, n_star=0, has_reciprocal_continuum=true)  # 4 stellar bases, it got confused
n_comp_by_eye!(df, 89; n_star=0, has_reciprocal_continuum=true)  # just looks hard, rvs correlate with score 1
n_comp_by_eye!(df, 90; n_star=0)  # just looks hard, rvs correlate with score 1
n_comp_by_eye!(df, 91; n_star=0)  # continuum basis vectors, this guess is totally unsupported by aic or rv
n_comp_by_eye!(df, 92; n_tel=2, n_star=0)  # wide continuum basis vectors
# n_comp_by_eye!(df, 93)  # just utterly screwed dont even bother
# n_comp_by_eye!(df, 94)  # just utterly screwed dont even bother
n_comp_by_eye!(df, 97; n_star=0)  # continuum basis vectors
n_comp_by_eye!(df, 98; n_tel=3, n_star=0, has_reciprocal_continuum=true)  # continuum basis vectors, aic and rv suggest more tel bases
# n_comp_by_eye!(df, 99; n_star=1)  # 5 stellar bases, it got confused, but with NaINIR?
n_comp_by_eye!(df, 99; n_tel=1, n_star=0)  # dont even bother trying to get NaINIR
n_comp_by_eye!(df, 100; n_star=0)  # rvs correlate with score 1
n_comp_by_eye!(df, 101; n_star=0, has_reciprocal_continuum=true)  # rvs correlate with score 1? aic says could add 1-2 more tel bases, bic says to only use 1
n_comp_by_eye!(df, 103; n_star=1)  # only use 1 basis for CaIRT3
n_comp_by_eye!(df, 105; n_tel=2, n_star=0, has_reciprocal_continuum=true)  # only use 1 basis for CaIRT3
n_comp_by_eye!(df, 106; n_star=1)  # 5 stellar bases, it got confused, probably screwed
n_comp_by_eye!(df, 107; n_star=1)  # 3 stellar bases, it got confused, probably screwed
# 108 and 109 borked
n_comp_by_eye!(df, 110; n_star=1)  # 5 stellar bases, it got confused, probably screwed
n_comp_by_eye!(df, 115; n_star=0)  # wide continuum basis
# 111 no data

# edge effect bases
n_comp_by_eye!(df, 31; n_star=0)  # doesn't affect rvs much
n_comp_by_eye!(df, 38; n_star=0)
n_comp_by_eye!(df, 41; n_star=0)  # doesn't affect rvs much
n_comp_by_eye!(df, 47; n_star=0)  # doesn't affect rvs much
n_comp_by_eye!(df, 48; n_star=0)  # doesn't affect rvs much
n_comp_by_eye!(df, 60; n_tel=0, n_star=0)  # doesn't affect rvs much
n_comp_by_eye!(df, 61; n_star=0)
n_comp_by_eye!(df, 95; n_star=0)
CSV.write("NEID/n_comps.csv", df)
