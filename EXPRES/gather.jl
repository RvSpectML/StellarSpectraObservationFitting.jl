## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

# Pkg.develop(;path="D:\\Christian\\Documents\\GitHub\\EMPCA")
# Pkg.develop(;path="C:\\Users\\Christian\\Dropbox\\GP_research\\julia\\StellarSpectraObservationFitting")
# Pkg.add(;url="https://github.com/christiangil/RvSpectMLBase.jl")
# Pkg.add(;url="https://github.com/christiangil/EchelleInstruments.jl")
# Pkg.add(;url="https://github.com/RvSpectML/RvSpectML.jl")

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

stars = ["10700", "26965"]
orders_list = [42:77, 40:77]
include("data_locs.jl")  # defines expres_data_path and expres_save_path

function retrieve(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/results.jld2" model rv_errors
    rvs_notel_opt = SSOF.rvs(model)
    return rvs_notel_opt, rv_errors
end

function retrieve_md(order::Int, star::String)
    @load expres_save_path*star*"/$(order)/model_decision.jld2" comp_ls ℓ aic bic ks test_n_comp_tel test_n_comp_star
    ans_aic = argmin(aic)
    ans_bic = argmin(bic)
    n_comps = [test_n_comp_tel[ans_aic[1]], test_n_comp_star[ans_aic[2]]]
    n_comps_bic = [test_n_comp_tel[ans_bic[1]], test_n_comp_star[ans_bic[2]]]
    return n_comps, n_comps_bic, ans_aic==ans_bic
end

# star_ind = SSOF.parse_args(1, Int, 2)
for star_ind in 1:2
    star = stars[star_ind]
    orders = orders_list[star_ind]

    @load expres_save_path*star*"/$(orders[1])/data.jld2" n_obs times_nu airmasses
    n_ord = length(orders)
    rvs = zeros(n_ord,  n_obs)
    rvs_σ = zeros(n_ord, n_obs)
    n_comps = zeros(Int, n_ord, 2)
    n_comps_bic = zeros(Int, n_ord, 2)
    robust = zeros(Bool, n_ord)
    for i in 1:n_ord
        try
            rvs[i, :], rvs_σ[i, :] = retrieve(orders[i], star)
            n_comps[i, :], n_comps_bic[i, :], robust[i] = retrieve_md(orders[i], star)
        catch
            rvs_σ[i, :] .= Inf
            n_comps[i, :] .= -1
            println("order $(orders[i]) is missing")
        end
    end

    @save "$(star)_rvs.jld2" rvs rvs_σ n_obs times_nu airmasses n_ord
    @save "$(star)_md.jld2" n_comps n_comps_bic robust
end
