## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

input_ind = SSOF.parse_args(1, Int, 0)

dates = ["2021/12/10", "2021/12/19", "2021/12/20", "2021/12/23"]

input_ind == 0 ? date_inds = (1:1) : date_inds = input_ind
orders_list = [4:122]
include("data_locs.jl")  # defines neid_data_path and neid_save_path
# prep_str = "noreg_"
prep_str = ""

function retrieve(order::Int, date::String)
    @load neid_save_path*date*"/$(order)/$(prep_str)results.jld2" model rv_errors
    rvs_notel_opt = SSOF.rvs(model)
    return rvs_notel_opt, rv_errors
end

function retrieve_md(order::Int, date::String)
    @load neid_save_path*date*"/$(order)/$(prep_str)model_decision.jld2" comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star
    ans_aic = argmin(aics)
    ans_bic = argmin(bics)
    n_comps = [test_n_comp_tel[ans_aic[1]], test_n_comp_star[ans_aic[2]]]
    n_comps_bic = [test_n_comp_tel[ans_bic[1]], test_n_comp_star[ans_bic[2]]]
    return n_comps, n_comps_bic, ans_aic==ans_bic
end

function retrieve_lcrvs(order::Int, date::String)
    @load neid_save_path*date*"/$(order)/$(prep_str)low_comp_rvs.jld2" rvs rvs_σ test_n_comp_tel test_n_comp_star
    return rvs, rvs_σ
end

reg_keys = SSOF._key_list[1:end-1]
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
        catch
            rvs_σ[i, :] .= Inf
            println("order $(orders[i]) is missing")
        end
    end
    @save "neid_$(prep_str)$(date)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord

    @load neid_save_path*date*"/$(orders[1])/data.jld2" n_obs times_nu airmasses
    n_comps = zeros(Int, n_ord, 2)
    n_comps_bic = zeros(Int, n_ord, 2)
    robust = zeros(Bool, n_ord)
    for i in 1:n_ord
        try
            n_comps[i, :], n_comps_bic[i, :], robust[i] = retrieve_md(orders[i], date)
        catch
            n_comps[i, :] .= -1
            println("order $(orders[i]) is missing")
        end
    end
    @save "neid_$(prep_str)$(date)_md.jld2" n_comps n_comps_bic robust

    # rvs = zeros(n_ord, 3, 3, n_obs)
    # rvs_σ = zeros(n_ord, 3, 3, n_obs)
    # for i in 1:n_ord
    #     try
    #         rvs[i, :, :, :], rvs_σ[i, :, :, :] = retrieve_lcrvs(orders[i], date)
    #     catch
    #         rvs_σ[i, :, :, :] .= Inf
    #         println("order $(orders[i]) is missing")
    #     end
    # end
    # @save "neid_$(prep_str)$(date)_lcrvs.jld2" rvs rvs_σ

    reg_tels = zeros(n_ord, length(reg_keys))
    reg_stars = zeros(n_ord, length(reg_keys))
    for i in 1:n_ord
        try
            reg_tels[i, :], reg_stars[i, :] = retrieve_reg(orders[i], date)
        catch
            println("order $(orders[i]) is missing")
        end
    end
    @save "neid_$(prep_str)$(date)_regs.jld2" reg_tels reg_stars
end
