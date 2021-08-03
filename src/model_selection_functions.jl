function test_ℓ_for_n_comps(n_comps::Vector, om::OrderModel, d::Data; return_inters::Bool=false, kwargs...)
    ws, l = WorkspaceTelStar(downsize(om, n_comps[1], n_comps[2]), d; return_loss_f=true)
    train_OrderModel!(ws; kwargs...)  # 16s
    train_OrderModel!(ws; g_tol=_g_tol_def/10*sqrt(length(ws.telstar.p0)), f_tol=1e-8, kwargs...)  # 50s
    if return_inters
        return ws, l, l(), (length(ws.telstar.p0) + length(ws.rv.p0))
    else
        return l(), (length(ws.telstar.p0) + length(ws.rv.p0))
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
        return n_comps, aic, bic
    else
        return n_comps
    end
end
