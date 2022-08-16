# functions related to calculating the PCA scores of time series spectra
using LinearAlgebra

"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/deriv_spectra_simple.jl
Estimate the derivatives of a vector
"""
function calc_deriv_RVSKL(x::Vector{<:Real})
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
function calc_doppler_component_RVSKL(lambda::Vector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(lambda) == length(flux)
	dlambdadpix = calc_deriv_RVSKL(lambda)
    dfluxdpix = calc_deriv_RVSKL(flux)
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end
function calc_doppler_component_RVSKL(lambda::Vector{T}, flux::Matrix{T}, kwargs...) where {T<:Real}
    return calc_doppler_component_RVSKL(lambda, vec(mean(flux, dims=2)), kwargs...)
end

function calc_doppler_component_RVSKL_Flux(lambda::Vector{T}, flux::Vector{T}) where {T<:Real}
    @assert length(lambda) == length(flux)
    dlambdadpix = calc_deriv_RVSKL_Flux(lambda)
    dfluxdpix = calc_deriv_RVSKL_Flux(flux)
    return dfluxdpix .* (lambda ./ dlambdadpix)  # doppler basis
end
function calc_doppler_component_RVSKL_Flux(lambda::Vector{T}, flux::Matrix{T}, kwargs...) where {T<:Real}
    return calc_doppler_component_RVSKL_Flux(lambda, vec(mean(flux, dims=2)), kwargs...)
end


function project_doppler_comp!(M::AbstractMatrix, scores::AbstractMatrix, Xtmp::AbstractMatrix, fixed_comp::AbstractVector)
	M[:, 1] = fixed_comp  # Force fixed (i.e., Doppler) component to replace first PCA component
	fixed_comp_norm2 = sum(abs2, fixed_comp)
	for i in 1:size(Xtmp, 2)
		scores[1, i] = dot(view(Xtmp, :, i), fixed_comp) / fixed_comp_norm2  # Normalize differently, so scores are z (i.e., doppler shift)
		Xtmp[:, i] -= scores[1, i] * fixed_comp
	end

	# calculating radial velocities (in m/s) from redshifts
	# I have no idea why the negative sign needs to be here
	rvs = -light_speed_nu * scores[1, :]  # c * z
	return rvs
end

function DEMPCA!(M::AbstractMatrix, scores::AbstractMatrix, μ::AbstractVector, Xtmp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; kwargs...) where {T<:Real}
	Xtmp .-= μ
	rvs = project_doppler_comp!(M, scores, Xtmp, doppler_comp)
	Xtmp .+= μ
	if size(M, 2) > 1
		EMPCA!(M, scores, μ, Xtmp, weights; inds=2:size(M, 2), kwargs...)
	end
	return rvs
end

_fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat; var_tot=sum(abs2, X)) =
	sum(abs2, X - Y) / var_tot
_fracvar(X::AbstractVecOrMat, Y::AbstractVecOrMat, weights::AbstractVecOrMat; var_tot=sum(abs2, X .* weights)) =
	sum(abs2, (X - Y) .* weights) / var_tot
function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat)
	var_tot = sum(abs2, X)
	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :); var_tot=var_tot) for i in 1:size(M, 2)]
end
function fracvar(X::AbstractVecOrMat, M::AbstractVecOrMat, s::AbstractVecOrMat, weights::AbstractVecOrMat)
	var_tot = sum(abs2, X .* weights)
	return [_fracvar(X, view(M, :, 1:i) * view(s, 1:i, :), weights; var_tot=var_tot) for i in 1:size(M, 2)]
end
