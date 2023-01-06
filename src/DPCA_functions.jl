# functions related to computing the Doppler-constrained PCA for time series spectra
using LinearAlgebra
import ExpectationMaximizationPCA as EMPCA
"""
Estimate the derivatives of a vector
"""
function simple_derivative(x::AbstractVector{<:Real})
    @assert length(x)>=3
    dx = similar(x)
    dx[1] = x[2]-x[1]
    dx[end] = x[end]-x[end-1]
	# faster than dx[2:end-1] .= (x[3:end] - x[1:end-2]) ./ 2
    for i in 2:(length(x)-1)
        dx[i] = (x[i+1]-x[i-1])/2
    end
    return dx
end

function simple_derivative_AD(x::Vector{<:Real})
	function helper(x::Vector, i::Int)
		if i == 1
			return x[2]-x[1]
		elseif i == length(x)
			return x[end]-x[end-1]
		else
			return (x[i+1]-x[i-1])/2
		end
	end
    @assert length(x)>=3
	return [helper(x, i) for i in eachindex(x)]
end


"""
Estimate the basis vector that encodes the effects of a doppler shift based on Taylor expanding f(λ/(1 + z)) about z=0
doppler_comp = λ * dF/dλ -> units of flux
"""
function doppler_component(λ::AbstractVector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(λ) == length(flux)
	dλdpix = simple_derivative(λ)
    dfluxdpix = simple_derivative(flux)
    return dfluxdpix .* (λ ./ dλdpix)  # doppler basis
end
doppler_component(λ::AbstractVector{T}, flux::Matrix{T}, kwargs...) where {T<:Real} = 
	doppler_component(λ, vec(mean(flux, dims=2)), kwargs...)
doppler_component_log(λ::AbstractVector{T}, flux::Vector{T}) where {T<:Real} = 
	doppler_component(λ, flux) ./ flux

function doppler_component_AD(λ::Vector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(λ) == length(flux)
    dλdpix = simple_derivative_AD(λ)
    dfluxdpix = simple_derivative_AD(flux)
    return dfluxdpix .* (λ ./ dλdpix)  # doppler basis
end
doppler_component_AD(λ::Vector{T}, flux::Matrix{T}, kwargs...) where {T<:Real} = 
	doppler_component_AD(λ, vec(mean(flux, dims=2)), kwargs...)
doppler_component_log_AD(λ::AbstractVector{T}, flux::Vector{T}) where {T<:Real} = 
	doppler_component_log_AD(λ, flux) ./ flux

function project_doppler_comp!(scores::AbstractVector, data_temp::AbstractMatrix, doppler_comp::AbstractVector, weights::AbstractMatrix)
	EMPCA._solve_scores!(doppler_comp, scores, data_temp, weights)
	data_temp .-= doppler_comp * scores'
	rvs = -light_speed_nu * scores  # c * z
	return rvs
end
function project_doppler_comp!(M::AbstractMatrix, scores::AbstractVector, data_temp::AbstractMatrix, doppler_comp::AbstractVector, weights::AbstractMatrix)
	M[:, 1] = doppler_comp  # Force fixed (i.e., Doppler) component to replace first PCA component
	return project_doppler_comp!(scores, data_temp, doppler_comp, weights)
end
function project_doppler_comp(data_temp::AbstractMatrix, doppler_comp::AbstractVector, weights::AbstractMatrix)
	scores = Array{Float64}(undef, size(weights, 2))
	return project_doppler_comp!(scores, data_temp, doppler_comp, weights)
end

function DEMPCA!(M::AbstractVecOrMat, scores::AbstractVecOrMat, rv_scores::AbstractVector, μ::AbstractVector, data_temp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; min_flux::Real=_min_flux_default, max_flux::Real=_max_flux_default, save_doppler_in_M1::Bool=true, kwargs...) where {T<:Real}
	data_temp .-= μ
	if save_doppler_in_M1
		rvs = project_doppler_comp!(M, rv_scores, data_temp, doppler_comp, weights)
	else
		rvs = project_doppler_comp!(rv_scores, data_temp, doppler_comp, weights)
	end
	data_temp .+= μ
	mask_low_pixels!(data_temp, weights; min_flux=min_flux, using_weights=true)
	mask_high_pixels!(data_temp, weights; max_flux=max_flux, using_weights=true)
	if size(M, 2) > 0
		EMPCA.EMPCA!(M, scores, μ, data_temp, weights; kwargs...)
	end
	return rvs
end
DEMPCA!(M::AbstractVecOrMat, scores::AbstractMatrix, μ::AbstractVector, data_temp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; inds=2:size(M, 2), kwargs...) where {T<:Real} =
	DEMPCA!(M, scores, view(scores, 1, :), μ, data_temp, weights, doppler_comp; inds=inds, kwargs...)
DEMPCA!(lm::FullLinearModel, data_temp, weights, doppler_comp; use_log=log_lm(lm), kwargs...) = DEMPCA!(lm.M, lm.s, lm.μ, data_temp, weights, doppler_comp; use_log=use_log, kwargs...)
DEMPCA!(lm::FullLinearModel, rv_scores, data_temp, weights, doppler_comp; use_log=log_lm(lm), kwargs...) = DEMPCA!(lm.M, lm.s, rv_scores, lm.μ, data_temp, weights, doppler_comp; use_log=use_log, kwargs...)

# _fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat; var_tot=sum(abs2, X)) =
# 	sum(abs2, X - Y) / var_tot
# _fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat, weights::AbstractVecOrMat; var_tot=sum(abs2, X .* weights)) =
# 	sum(abs2, (X - Y) .* weights) / var_tot
# function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat)
# 	var_tot = sum(abs2, X)
# 	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :); var_tot=var_tot) for i in axes(M, 2)]
# end
# function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat, weights::AbstractVecOrMat)
# 	var_tot = sum(abs2, X .* weights)
# 	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :), weights; var_tot=var_tot) for i in axes(M, 2)]
# end
