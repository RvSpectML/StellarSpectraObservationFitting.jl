# functions related to calculating the PCA scores of time series spectra
using LinearAlgebra

"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Estimate the derivatives of a vector
"""
function calc_deriv_RVSKL(x::AbstractVector{<:Real})
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

function calc_deriv_RVSKL_Flux(x::Vector{<:Real})
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
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Function to estimate the derivative(s) of the mean spectrum
doppler_comp = λ * dF/dλ -> units of flux
"""
function calc_doppler_component_RVSKL(lambda::AbstractVector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(lambda) == length(flux)
	dlambdadpix = calc_deriv_RVSKL(lambda)
    dfluxdpix = calc_deriv_RVSKL(flux)
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end
function calc_doppler_component_RVSKL(lambda::AbstractVector{T}, flux::Matrix{T}, kwargs...) where {T<:Real}
    return calc_doppler_component_RVSKL(lambda, vec(mean(flux, dims=2)), kwargs...)
end
calc_doppler_component_RVSKL_log(lambda::AbstractVector{T}, flux::Vector{T}) where {T<:Real} = 
	calc_doppler_component_RVSKL(lambda, flux) ./ flux

function calc_doppler_component_RVSKL_Flux(lambda::Vector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(lambda) == length(flux)
    dlambdadpix = calc_deriv_RVSKL_Flux(lambda)
    dfluxdpix = calc_deriv_RVSKL_Flux(flux)
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end
function calc_doppler_component_RVSKL_Flux(lambda::Vector{T}, flux::Matrix{T}, kwargs...) where {T<:Real}
    return calc_doppler_component_RVSKL_Flux(lambda, vec(mean(flux, dims=2)), kwargs...)
end

# function project_doppler_comp!(M::AbstractMatrix, scores::AbstractVector, Xtmp::AbstractMatrix, fixed_comp::AbstractVector)
# 	M[:, 1] = fixed_comp  # Force fixed (i.e., Doppler) component to replace first PCA component
# 	fixed_comp_norm2 = sum(abs2, fixed_comp)
# 	for i in axes(Xtmp, 2)
# 		scores[i] = dot(view(Xtmp, :, i), fixed_comp) / fixed_comp_norm2  # Normalize differently, so scores are z (i.e., doppler shift)
# 		Xtmp[:, i] -= scores[i] * fixed_comp
# 	end
#
# 	# calculating radial velocities (in m/s) from redshifts
# 	# I have no idea why the negative sign needs to be here
# 	rvs = -light_speed_nu * scores  # c * z
# 	return rvs
# end

function project_doppler_comp!(scores::AbstractVector, Xtmp::AbstractMatrix, fixed_comp::AbstractVector, weights::AbstractMatrix)
	_solve_coeffs!(fixed_comp, scores, Xtmp, weights)
	Xtmp .-= fixed_comp * scores'
	rvs = -light_speed_nu * scores  # c * z
	return rvs
end
function project_doppler_comp!(M::AbstractMatrix, scores::AbstractVector, Xtmp::AbstractMatrix, fixed_comp::AbstractVector, weights::AbstractMatrix)
	M[:, 1] = fixed_comp  # Force fixed (i.e., Doppler) component to replace first PCA component
	return project_doppler_comp!(scores, Xtmp, fixed_comp, weights)
end
function project_doppler_comp(Xtmp::AbstractMatrix, fixed_comp::AbstractVector, weights::AbstractMatrix)
	scores = Array{Float64}(undef, size(weights, 2))
	return project_doppler_comp!(scores, Xtmp, fixed_comp, weights)
end

function DEMPCA!(M::AbstractVecOrMat, scores::AbstractVecOrMat, rv_scores::AbstractVector, μ::AbstractVector, Xtmp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; min_flux::Real=_min_flux_default, max_flux::Real=_max_flux_default, save_doppler_in_M1::Bool=true, kwargs...) where {T<:Real}
	Xtmp .-= μ
	if save_doppler_in_M1
		rvs = project_doppler_comp!(M, rv_scores, Xtmp, doppler_comp, weights)
	else
		rvs = project_doppler_comp!(rv_scores, Xtmp, doppler_comp, weights)
	end
	Xtmp .+= μ
	mask_low_pixels!(Xtmp, weights; min_flux=min_flux, using_weights=true)
	mask_high_pixels!(Xtmp, weights; max_flux=max_flux, using_weights=true)
	if size(M, 2) > 0
		EMPCA!(M, scores, μ, Xtmp, weights; kwargs...)
	end
	return rvs
end
DEMPCA!(M::AbstractVecOrMat, scores::AbstractMatrix, μ::AbstractVector, Xtmp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; inds=2:size(M, 2), kwargs...) where {T<:Real} =
	DEMPCA!(M, scores, view(scores, 1, :), μ, Xtmp, weights, doppler_comp; inds=inds, kwargs...)
DEMPCA!(lm::FullLinearModel, Xtmp, weights, doppler_comp; log_lm=log_lm(lm), kwargs...) = DEMPCA!(lm.M, lm.s, lm.μ, Xtmp, weights, doppler_comp; log_lm=log_lm, kwargs...)
DEMPCA!(lm::FullLinearModel, rv_scores, Xtmp, weights, doppler_comp; log_lm=log_lm(lm), kwargs...) = DEMPCA!(lm.M, lm.s, rv_scores, lm.μ, Xtmp, weights, doppler_comp; log_lm=log_lm, kwargs...)

_fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat; var_tot=sum(abs2, X)) =
	sum(abs2, X - Y) / var_tot
_fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat, weights::AbstractVecOrMat; var_tot=sum(abs2, X .* weights)) =
	sum(abs2, (X - Y) .* weights) / var_tot
function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat)
	var_tot = sum(abs2, X)
	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :); var_tot=var_tot) for i in axes(M, 2)]
end
function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat, weights::AbstractVecOrMat)
	var_tot = sum(abs2, X .* weights)
	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :), weights; var_tot=var_tot) for i in axes(M, 2)]
end
