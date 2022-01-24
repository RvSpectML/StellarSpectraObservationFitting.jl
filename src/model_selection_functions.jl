total_length(x::Vector{<:AbstractArray}) = sum(total_length.(x))
total_length(x::AbstractArray) = length(x)
total_length(mws::TelStarWorkspace) = total_length(mws.telstar.θ) + total_length(mws.rv.θ)
total_length(mws::TotalWorkspace) = total_length(mws.total.θ)
total_length(mws::OptimWorkspace) = length(mws.telstar.p0) + length(mws.rv.p0)

function test_ℓ_for_n_comps(n_comps::Vector, mws_inp::ModelWorkspace; return_inters::Bool=false, kwargs...)
    mws = typeof(mws_inp)(downsize(mws_inp.om, n_comps[1], n_comps[2]), mws_inp.d)
    l = loss_func(mws)
    fine_train_OrderModel!(mws; kwargs...)  # 16s
    if return_inters
        return mws, l, l(), total_length(mws), std(rvs(mws.om))
    else
        return l(), total_length(mws), std(rvs(mws.om))
    end
end

function choose_n_comps(ls::Matrix, ks::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector, var::AbstractMatrix; return_inters::Bool=false)

    ## max likelihood
    # ans_ml = argmin(ls)

    mask = var .!= Inf
    n = sum(mask)
    ℓ = -1/2 .* (ls .+ (sum(log.(var[mask])) + (n * log(2 * π))))
    aic = 2 .* (ks - ℓ)
    ans_aic = argmin(aic)

    bic = log(n) .* ks - 2 .* ℓ
    ans_bic = argmin(bic)

    if ans_aic != ans_bic; @warn "AIC and BIC gave different answers" end

    n_comps = [test_n_comp_tel[ans_aic[1]], test_n_comp_star[ans_aic[2]]]
    if return_inters
        return n_comps, ℓ, aic, bic
    else
        return n_comps
    end
end
