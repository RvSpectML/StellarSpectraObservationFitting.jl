total_length(x::Vector{<:AbstractArray}) = sum(total_length.(x))
total_length(x::AbstractArray) = length(x)
"""
    total_length(mws)

Calculates the number of parameters of `mws`
"""
total_length(mws::AdamWorkspace) = total_length(mws.total.θ)
total_length(mws::OptimTelStarWorkspace) = length(mws.telstar.p0) + length(mws.rv.p0)
total_length(mws::OptimTotalWorkspace) = length(mws.total.p0)


"""
    intra_night_std(rvs, times; thres=3, show_warn=true)

Calculates the intra-night std for `rvs` time series observed at `times` (in days)
"""
function intra_night_std(rvs::AbstractVector, times::AbstractVector; thres::Int=3, show_warn::Bool=true)
    intra_night_stds = [std(rvs[i]) for i in observation_night_inds(times) if length(i)>(thres-1)]
    if length(intra_night_stds) < 1
        if show_warn; @warn "no nights to base the intra night std of the RVs on. Returning the std of all of the observations" end
        return Inf
    elseif length(intra_night_stds) < 2
        if show_warn; @warn "only one night to base the intra night std of the RVs on" end
    elseif length(intra_night_stds) < 3
        if show_warn; @warn "only a couple of nights to base the intra night std of the RVs on" end
    end
    return median(intra_night_stds)
end

n_negligible(x::AbstractVecOrMat) = sum(abs.(x) .< (1e-5 * sqrt(sum(abs2, x))))
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


# function _test_om(mws_inp::ModelWorkspace, om::OrderModel, times::AbstractVector; no_tels::Bool=false, kwargs...)
#     if no_tels
# 		mws = FrozenTelWorkspace(om, mws_inp.d)
# 		om.tel.lm.μ .= 1
# 	else
# 		mws = typeof(mws_inp)(om, mws_inp.d)
# 	end
#     train_OrderModel!(mws; kwargs...)  # 16s
#     n = total_length(mws) #- n_negligible(mws)
#     model_rvs = rvs(mws.om)
#     return _loss(mws), n, std(model_rvs), intra_night_std(model_rvs, times; show_warn=false)
# end
# function test_ℓ_for_n_comps(n_comps::Vector, mws_inp::ModelWorkspace, times::AbstractVector, lm_tel::Vector{<:LinearModel}, lm_star::Vector{<:LinearModel}; return_inters::Bool=false, lm_tel_ind::Int=n_comps[2]+1, lm_star_ind::Int=n_comps[1]+1, kwargs...)
#     _om = downsize(mws_inp.om, max(0, n_comps[1]), n_comps[2])

#     # if either of the models are constant, there will only be one initialization
#     # that should already be stored in the model
#     if (n_comps[1] <= 0) || (n_comps[2] == 0)
#         # if n_comps[2] > 0; fill_StarModel!(_om, lm_star[1]; inds=(1:n_comps[2]) .+ 1) end
#         l, n, rv_std, in_rv_std = _test_om(mws_inp, _om, times; no_tels=n_comps[1]<0, kwargs...)
#         return l, n, rv_std, in_rv_std, 2

#     # choose the better of the two initializations
#     else
#         ls = zeros(2)
#         ns = zeros(Int, 2)
#         rv_stds = zeros(2)
#         in_rv_stds = zeros(2)

#         # test telluric components first
# 		_fill_model!(_om, n_comps, 1, lm_tel, lm_star; lm_tel_ind=lm_tel_ind, lm_star_ind=lm_star_ind)
#         ls[1], ns[1], rv_stds[1], in_rv_stds[1] = _test_om(mws_inp, _om, times; kwargs...)

#         # test star components next
# 		_fill_model!(_om, n_comps, 2, lm_tel, lm_star; lm_tel_ind=lm_tel_ind, lm_star_ind=lm_star_ind)
#         ls[2], ns[2], rv_stds[2], in_rv_stds[2] = _test_om(mws_inp, _om, times; kwargs...)

#         better_model = argmin(ls)
#         return ls[better_model], ns[better_model], rv_stds[better_model], in_rv_stds[better_model], better_model
#     end
# end
# function _fill_model!(model::OrderModel, n_comps::Vector{<:Int}, better_model::Int, lm_tels::Vector{<:LinearModel}, lm_stars::Vector{<:LinearModel}; lm_tel_ind::Int=n_comps[2]+1, lm_star_ind::Int=n_comps[1]+1)
# 	# if all(n_comps .> 0)
# 	@assert better_model in [1,2]
# 	if better_model == 1
# 		lm_tel = lm_tels[1]
# 		lm_star = lm_stars[lm_star_ind]
# 	else
# 		lm_tel = lm_tels[lm_tel_ind]
# 		lm_star = lm_stars[1]
# 	end
# 	fill_TelModel!(model, lm_tel, 1:n_comps[1])
# 	fill_StarModel!(model, lm_star; inds=(1:n_comps[2]) .+ 1)
# 	# end
# end


"""
    ℓ_prereqs(vars)

Calculate some terms needed to calculate the log-likelihood
"""
function ℓ_prereqs(vars::Matrix)
	mask = isfinite.(vars)
	n = sum(mask)
	logdet_Σ = sum(log.(vars[mask]))
	return logdet_Σ, n
end
"Gaussian log-likelihood function"
ℓ(χ²::Real, logdet_Σ::Real, n::Int) = -1/2 * (χ² + logdet_Σ + n * _log2π)
"Akaike information criterion"
aic(k::Int, ℓ::Real) = 2 * (k - ℓ)
aic(mws::ModelWorkspace, logdet_Σ::Real, n::Int) =
	aic(total_length(mws), ℓ(_loss(mws), logdet_Σ, n))
function aic(mws::ModelWorkspace)
	n, logdet_Σ = ℓ_prereqs(mws.d.var)
	return aic(mws, logdet_Σ, n)
end
"Akaike information criterion (corrected for small sample sizes)"
aicc(k::Int, ℓ::Real, n::Int) = aic(k, ℓ) + (2k(k+1))/(n-k-1)
aicc(mws::ModelWorkspace, logdet_Σ::Real, n::Int) =
	aicc(total_length(mws), ℓ(_loss(mws), logdet_Σ, n), n)
function aicc(mws::ModelWorkspace)
	n, logdet_Σ = ℓ_prereqs(mws.d.var)
	return aicc(mws, logdet_Σ, n)
end
"Bayesian information criterion"
bic(k::Int, ℓ::Real, n::Int) = k * log(n) - 2 * ℓ

# function choose_n_comps(ls::Matrix, ks::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector, var::AbstractMatrix; return_inters::Bool=false, use_aic::Bool=true)

#     ## max likelihood
#     # ans_ml = argmin(ls)

# 	n, logdet_Σ = ℓ_prereqs(var)

#     ℓs = ℓ.(ls, logdet_Σ, n)
# 	ℓs[isnan.(ℓs)] .= -Inf
#     aics = aic.(ks, ℓs)
#     ans_aic = argmin(aics)

#     bics = bic.(ks, ℓs, n)
#     ans_bic = argmin(bics)

#     if ans_aic != ans_bic; @warn "AIC and BIC gave different answers" end

#     use_aic ? best_ind = ans_aic : best_ind = ans_bic
#     n_comps = [test_n_comp_tel[best_ind[1]], test_n_comp_star[best_ind[2]]]
#     if return_inters
#         return n_comps, ℓs, aics, bics, best_ind
#     else
#         return n_comps
#     end
# end
