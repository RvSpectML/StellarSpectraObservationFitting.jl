total_length(x::Vector{<:AbstractArray}) = sum(total_length.(x))
total_length(x::AbstractArray) = length(x)
total_length(mws::AdamWorkspace) = total_length(mws.total.θ)
total_length(mws::OptimTelStarWorkspace) = length(mws.telstar.p0) + length(mws.rv.p0)
total_length(mws::OptimTotalWorkspace) = length(mws.total.p0)
function effective_length(x; return_mask::Bool=false, masked_val::Real = Inf)
    mask = x .!= masked_val
    if return_mask
        return sum(mask), mask
    else
        return sum(mask)
    end
end
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

function _test_om(mws_inp::ModelWorkspace, om::OrderModel, times::AbstractVector; no_tels::Bool=false, kwargs...)
    if no_tels
		mws = FrozenTelWorkspace(om, mws_inp.d)
		om.tel.lm.μ .= 1
	else
		mws = typeof(mws_inp)(om, mws_inp.d)
	end
    train_OrderModel!(mws; kwargs...)  # 16s
    n = total_length(mws) #- n_negligible(mws)
    model_rvs = rvs(mws.om)
    return _loss(mws), n, std(model_rvs), intra_night_std(model_rvs, times; show_warn=false)
end
function test_ℓ_for_n_comps(n_comps::Vector, mws_inp::ModelWorkspace, times::AbstractVector, lm_tel::Vector{<:LinearModel}, lm_star::Vector{<:LinearModel}; return_inters::Bool=false, kwargs...)
    _om = downsize(mws_inp.om, max(0, n_comps[1]), n_comps[2])

    # if either of the models are constant, there will only be one initialization
    # that should already be stored in the model
    if (n_comps[1] <= 0) || (n_comps[2] == 0)
        # if n_comps[2] > 0; fill_StarModel!(_om, lm_star[1]; inds=(1:n_comps[2]) .+ 1) end
        l, n, rv_std, in_rv_std = _test_om(mws_inp, _om, times; no_tels=n_comps[1]<0, kwargs...)
        return l, n, rv_std, in_rv_std, 2

    # choose the better of the two initializations
    else
        ls = zeros(2)
        ns = zeros(Int, 2)
        rv_stds = zeros(2)
        in_rv_stds = zeros(2)

        # test telluric components first
		_fill_model!(_om, n_comps, 1, lm_tel, lm_star)
        ls[1], ns[1], rv_stds[1], in_rv_stds[1] = _test_om(mws_inp, _om, times; kwargs...)

        # test star components next
		_fill_model!(_om, n_comps, 2, lm_tel, lm_star)
        ls[2], ns[2], rv_stds[2], in_rv_stds[2] = _test_om(mws_inp, _om, times; kwargs...)

        better_model = argmin(ls)
        return ls[better_model], ns[better_model], rv_stds[better_model], in_rv_stds[better_model], better_model
    end
end
function _fill_model!(model::OrderModel, n_comps::Vector{<:Int}, better_model::Int, lm_tels::Vector{<:LinearModel}, lm_stars::Vector{<:LinearModel}; lm_tel_ind::Int=n_comps[2]+1, lm_star_ind::Int=n_comps[1]+1)
	# if all(n_comps .> 0)
	@assert better_model in [1,2]
	if better_model == 1
		lm_tel = lm_tels[1]
		lm_star = lm_stars[lm_star_ind]
	else
		lm_tel = lm_tels[lm_tel_ind]
		lm_star = lm_stars[1]
	end
	fill_TelModel!(model, lm_tel, 1:n_comps[1])
	fill_StarModel!(model, lm_star; inds=(1:n_comps[2]) .+ 1)
	# end
end

function choose_n_comps(ls::Matrix, ks::Matrix, test_n_comp_tel::AbstractVector, test_n_comp_star::AbstractVector, var::AbstractMatrix; return_inters::Bool=false, use_aic::Bool=true)

    ## max likelihood
    # ans_ml = argmin(ls)

    n, mask = effective_length(var; return_mask=true)
    ℓ = -1/2 .* (ls .+ (sum(log.(var[mask])) + (n * log(2 * π))))
	ℓ[isnan.(ℓ)] .= -Inf
    aic = 2 .* (ks - ℓ)
    ans_aic = argmin(aic)

    bic = log(n) .* ks - 2 .* ℓ
    ans_bic = argmin(bic)

    if ans_aic != ans_bic; @warn "AIC and BIC gave different answers" end

    use_aic ? best_ind = ans_aic : best_ind = ans_bic
    n_comps = [test_n_comp_tel[best_ind[1]], test_n_comp_star[best_ind[2]]]
    if return_inters
        return n_comps, ℓ, aic, bic, best_ind
    else
        return n_comps
    end
end
