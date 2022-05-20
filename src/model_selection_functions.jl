total_length(x::Vector{<:AbstractArray}) = sum(total_length.(x))
total_length(x::AbstractArray) = length(x)
total_length(mws::FrozenTelWorkspace) = total_length(mws.total.θ) - is_time_variable(mws.om.tel) * length(mws.total.θ[1][1]) - length(mws.total.θ[1][end])
total_length(mws::TotalWorkspace) = total_length(mws.total.θ)
total_length(mws::OptimWorkspace) = length(mws.telstar.p0) + length(mws.rv.p0)
function effective_length(x; return_mask::Bool=false, masked_val::Real = Inf)
    mask = x .!= masked_val
    if return_mask
        return sum(mask), mask
    else
        return sum(mask)
    end
end
function intra_night_std(rvs::Vector, times::Vector; thres::Int=3)
    intra_night_stds = [std(rvs[i]) for i in observation_night_inds(times) if length(i)>(thres-1)]
    if length(intra_night_stds) < 1
        @warn "no nights to base the intra night std of the RVs on. Returning the std of all of the observations"
        return std(rvs)
    elseif length(intra_night_stds) < 2
        @warn "only one night to base the intra night std of the RVs on"
    elseif length(intra_night_stds) < 3
        @warn "only a couple of nights to base the intra night std of the RVs on"
    end
    return median(intra_night_stds)
end

n_negligible(x::AbstractVecOrMat) = sum(abs.(x) .< (1e-4 * sqrt(sum(abs2, x))))
function n_negligible(x::Submodel)
    n = n_negligible(x.lm.μ)
    if is_time_variable(x); n += n_negligible(x.lm.M) end
    return n
end
function n_negligible(mws::ModelWorkspace)
    n = n_negligible(mws.om.star)
    if !(typeof(mws) <: FrozenTelWorkspace); n += n_negligible(mws.om.tel) end
    return n
end
function test_ℓ_for_n_comps_basic(n_comps::Vector, mws_inp::ModelWorkspace; return_inters::Bool=false, iter=50, kwargs...)
    mws = typeof(mws_inp)(downsize(mws_inp.om, n_comps[1], n_comps[2]), mws_inp.d)
    train_OrderModel!(mws; iter=iter, kwargs...)  # 16s
    n = total_length(mws) - n_negligible(mws)
    if return_inters
        return mws, _loss(mws), n
    end
    return _loss(mws), n
end
function test_ℓ_for_n_comps(n_comps::Vector, mws_inp::ModelWorkspace, times::Vector; return_inters::Bool=false, iter=50, kwargs...)
    mws, l, len = test_ℓ_for_n_comps_basic(n_comps, mws_inp; return_inters=true, iter=iter, kwargs...)
    model_rvs = rvs(mws.om)
    return l, len, std(model_rvs), intra_night_std(model_rvs, times)
end

function choose_n_comps(ls::Matrix, ks::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector, var::AbstractMatrix; return_inters::Bool=false, use_aic::Bool=true)

    ## max likelihood
    # ans_ml = argmin(ls)

    n, mask = effective_length(var; return_mask=true)
    ℓ = -1/2 .* (ls .+ (sum(log.(var[mask])) + (n * log(2 * π))))
    aic = 2 .* (ks - ℓ)
    ans_aic = argmin(aic)

    bic = log(n) .* ks - 2 .* ℓ
    ans_bic = argmin(bic)

    if ans_aic != ans_bic; @warn "AIC and BIC gave different answers" end

    use_aic ?
        n_comps = [test_n_comp_tel[ans_aic[1]], test_n_comp_star[ans_aic[2]]] :
        n_comps = [test_n_comp_tel[ans_bic[1]], test_n_comp_star[ans_bic[2]]]
    if return_inters
        return n_comps, ℓ, aic, bic
    else
        return n_comps
    end
end
