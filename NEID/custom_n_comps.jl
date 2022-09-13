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

