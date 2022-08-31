## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF

## Setting up necessary variables

input_ind = SSOF.parse_args(1, Int, 0)

dates = ["2021/12/10", "2021/12/19", "2021/12/20", "2021/12/23"]

input_ind == 0 ? date_inds = (1:4) : date_inds = input_ind
orders_list = repeat([4:118], length(stars))
include("data_locs.jl")  # defines neid_data_path and neid_save_path
# prep_str = "noreg_"
prep_str = "nolsf_"

function retrieve(order::Int, date::String)
    @load neid_save_path*date*"/$(order)/$(prep_str)results.jld2" model rv_errors
    rvs_notel_opt = SSOF.rvs(model)
    return rvs_notel_opt, rv_errors
end

reg_keys = [:GP_μ, :L1_μ, :L1_μ₊_factor, :GP_M, :L1_M]
function retrieve_reg(order::Int, date::String)
    @load neid_save_path*date*"/$(order)/$(prep_str)results.jld2" model
    return [model.reg_tel[k] for k in reg_keys], [model.reg_star[k] for k in reg_keys]
end

for date_ind in date_inds
    date = dates[date_ind]
    orders = orders_list[date_ind]
    n_ord = length(orders)
    @load neid_save_path*date*"/$(orders[1])/data.jld2" n_obs times_nu airmasses

    rvs = zeros(n_ord,  n_obs)
    rvs_σ = zeros(n_ord, n_obs)
    for i in 1:n_ord
        try
            rvs[i, :], rvs_σ[i, :] = retrieve(orders[i], date)
        catch err
            if isa(err, SystemError)
                rvs_σ[i, :] .= Inf
                println("order $(orders[i]) is missing")
            else
                rethrow()
            end
        end
    end
    @save neid_save_path*date*"/$(prep_str)rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

    reg_tels = zeros(n_ord, length(reg_keys))
    reg_stars = zeros(n_ord, length(reg_keys))
    for i in 1:n_ord
        try
            reg_tels[i, :], reg_stars[i, :] = retrieve_reg(orders[i], date)
        catch err
            if isa(err, SystemError)
                println("order $(orders[i]) is missing")
            else
                rethrow()
            end
        end
    end
    @save neid_save_path*date*"/$(prep_str)regs.jld2" reg_tels reg_stars
end
