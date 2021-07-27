## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

stars = ["10700", "26965"]
orders_list = [42:77, 40:77]
include("data_locs.jl")  # defines expres_data_path and expres_save_path

function retrieve(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/results.jld2" model rv_errors
    rvs_notel_opt = (model.rv.lm.s .* SSOF.light_speed_nu)'
    return rvs_notel_opt, rv_errors
end

# star_ind = SSOF.parse_args(1, Int, 2)
for star_ind in 1:2
    star = stars[star_ind]
    orders = orders_list[star_ind]

    @load expres_save_path*star*"/$(orders[1])/data.jld2" n_obs times_nu airmasses
    n_ord = length(orders)
    rvs = zeros(n_ord,  n_obs)
    rvs_σ = zeros(n_ord, n_obs)

    for i in 1:n_ord
        try
            rvs[i, :], rvs_σ[i, :] = retrieve(orders[i], star)
        catch
            rvs_σ[i, :] .= Inf
            println("order $(orders[i]) is missing")
        end
    end

    @save "$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
end
