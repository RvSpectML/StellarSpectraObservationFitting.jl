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
    @load fn model rvs_σ
    rvs_notel_opt = SSOF.rvs(model)
    if return_extra
        SSOF.is_time_variable(model.tel) ? n_comp_tel = size(model.tel.lm.M, 2) : n_comp_tel = 0
        SSOF.is_time_variable(model.star) ? n_comp_star = size(model.star.lm.M, 2) : n_comp_star = 0
        return rvs_notel_opt, rvs_σ, [n_comp_tel, n_comp_star], model.reg_tel, model.reg_star
    end
    return rvs_notel_opt, rvs_σ
end
retrieve_rvs(args...; kwargs...) = safe_retrieve(_retrieve_rvs, args...; kwargs...)

function _retrieve_md(fn::String)
    @load fn comp_ls ℓ aics bics ks test_n_comp_tel test_n_comp_star comp_stds comp_intra_stds
    return comp_ls, ℓ, aics, bics, ks, test_n_comp_tel, test_n_comp_star, comp_stds, comp_intra_stds
end
retrieve_md(args...; kwargs...) = safe_retrieve(_retrieve_md, args...; kwargs...)

function retrieve(n_obs::Int, rv_fns::Vector{String}, model_fns::Vector{String}, data_fns::Vector{String})
    n_ord = length(rv_fns)
    all_rvs = zeros(n_ord,  n_obs)
    all_rvs_σ = Inf .* ones(n_ord, n_obs)
    constant = trues(n_ord)
    no_tel = trues(n_ord)
    wavelength_range = Array{Float64}(undef, n_ord, 2)
    all_star_s = []
    all_tel_s = []

    function catch_f(err, fns, i)
        if isa(err, SystemError)
            println(fns[i] * " (orders[$i]) is missing")
        elseif isa(err, KeyError)
            println("orders[$i] analysis is incomplete")
        else
            rethrow(err)
        end
    end

    for i in 1:n_ord
        try
            @load rv_fns[i] rvs rvs_σ
            all_rvs[i, :] = rvs
            all_rvs_σ[i, :] = rvs_σ
        catch err
            catch_f(err, rv_fns, i)
        end
        try
            @load model_fns[i] model
            tel_var = SSOF.is_time_variable(model.tel)
            star_var = SSOF.is_time_variable(model.star)
            constant[i] = !(tel_var || star_var)
            no_tel[i] = all(isone.(model.tel.lm.μ)) && !tel_var
            if tel_var
                append!(all_tel_s, [model.tel.lm.s])
            else
                append!(all_tel_s, [[]])
            end
            if star_var
                append!(all_star_s, [model.star.lm.s])
            else
                append!(all_star_s, [[]])
            end
        catch err
            catch_f(err, model_fns, i)
            append!(all_tel_s, [[]])
            append!(all_star_s, [[]])
        end
        try
            @load data_fns[i] data
            wavelength_range[i, :] .= exp.(extrema(data.log_λ_star[.!(isinf.(data.var_s))]))
        catch err
            catch_f(err, data_fns, i)
        end
    end
    return all_rvs, all_rvs_σ, constant, no_tel, wavelength_range, all_star_s, all_tel_s
end

function retrieve(save_fns::Vector{String}, rv_fns::Vector{Vector{String}}, model_fns::Vector{Vector{String}}, data_fns::Vector{Vector{String}})
    @assert length(rv_fns) == length(save_fns) == length(data_fns) == length(model_fns)
    for i in 1:length(save_fns)
        @load data_fns[i][1] n_obs times_nu airmasses
        n_ord = length(rv_fns[i])
        rvs, rvs_σ, constant, no_tel, wavelength_range, all_star_s, all_tel_s = retrieve(n_obs, rv_fns[i], model_fns[i], data_fns[i])
        @save save_fns[i] n_obs times_nu airmasses n_ord rvs rvs_σ constant no_tel wavelength_range all_star_s all_tel_s
    end
end
