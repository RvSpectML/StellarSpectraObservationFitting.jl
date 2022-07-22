## Some helpful gathering functions
import StellarSpectraObservationFitting as SSOF
using JLD2

function safe_retrieve(retrieve_f, args...; pre_string="order", kwargs...)
    try
        return retrieve_f(args...; kwargs...)
    catch err
        if isa(err, SystemError)
            println(pre_string * " is missing")
        elseif isa(err, KeyError)
            println(pre_string * " analysis is incomplete")
        else
            rethrow()
        end
    end
end

function _retrieve_rvs(fn::String; return_extra::Bool=false)
    @load fn model rv_errors
    rvs_notel_opt = SSOF.rvs(model)
    if return_extra
        SSOF.is_time_variable(model.tel) ? n_comp_tel = size(model.tel.lm.M, 2) : n_comp_tel = 0
        SSOF.is_time_variable(model.star) ? n_comp_star = size(model.star.lm.M, 2) : n_comp_star = 0
        return rvs_notel_opt, rv_errors, [n_comp_tel, n_comp_star], model.reg_tel, model.reg_star
    end
    return rvs_notel_opt, rv_errors
end
retrieve_rvs(args...; kwargs...) = safe_retrieve(_retrieve_rvs, args...; kwargs...)

function _retrieve_md(fn::String)
    @load fn comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds
    return comp_ls, ℓ, aics, bics, ks, test_n_comp_tel, test_n_comp_star, comp_stds, comp_intra_stds
end
retrieve_md(args...; kwargs...) = safe_retrieve(_retrieve_md, args...; kwargs...)

function retrieve_all_rvs(n_obs::Int, fns::Vector{String})
    n_ord = length(fns)
    rvs = zeros(n_ord,  n_obs)
    rvs_σ = Inf .* ones(n_ord, n_obs)
    for i in 1:n_ord
        try
            rvs[i, :], rvs_σ[i, :] = _retrieve_rvs(fns[i])
        catch err
            if isa(err, SystemError)
                println(fns[i] * " (orders[$i]) is missing")
            elseif isa(err, KeyError)
                println("orders[$i] analysis is incomplete")
            else
                rethrow()
            end
        end
    end
    return rvs, rvs_σ
end

function retrieve_all_rvs(data_fns::Vector{String}, fns::Vector{Vector{String}}, save_fns::Vector{String})
    @assert length(fns) == length(save_fns) == length(data_fns)
    for i in 1:length(save_fns)
        @load data_fns[i] n_obs times_nu airmasses
        n_ord = length(fns[i])
        rvs, rvs_σ = retrieve_all_rvs(n_obs, fns[i])
        @save save_fns[i] rvs rvs_σ n_obs times_nu airmasses n_ord
    end
end
