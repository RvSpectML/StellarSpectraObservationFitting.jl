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

function EMPCA!(M::AbstractMatrix, scores::AbstractMatrix, Xtmp::AbstractMatrix, weights::AbstractMatrix; inds::UnitRange{<:Int}=1:size(M, 2), niter::Int=100)
	@assert inds[1] > 0
	_empca!(view(M, :, inds), view(scores, inds, :), Xtmp, weights, nvec=length(inds), niter=niter)
end

function DEMPCA!(spectra::Matrix{T}, λs::Vector{T}, M::AbstractMatrix, scores::AbstractMatrix, weights::Matrix{T};
	template::Vector{T}=make_template(spectra), kwargs...) where {T<:Real}
	doppler_comp = calc_doppler_component_RVSKL(λs, template)
    return DEMPCA!(M, scores, spectra .- template, weights, doppler_comp; kwargs...)
end

function DEMPCA!(M::AbstractMatrix, scores::AbstractMatrix, Xtmp::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; kwargs...) where {T<:Real}
	rvs = project_doppler_comp!(M, scores, Xtmp, doppler_comp)
	if size(M, 2) > 1
		EMPCA!(M, scores, Xtmp, weights; inds=2:size(M, 2), kwargs...)
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


## EMPCA implementation

function _solve_coeffs!(eigvec, coeff, data, weights)
	nobs = size(data, 2)
	for i in 1:nobs
		coeff[:, i] .= _solve(eigvec, view(data, :, i), view(weights, :, i))
	end
	# solve_model!(model, eigvec, coeff)
end

function _solve_eigenvectors!(eigvec, coeff, data, weights)
	nvar, nvec = size(eigvec)
	cw = Array{Float64}(undef, size(data, 2))
	for i in 1:nvec
		c = view(coeff, i, :)
		for j in 1:nvar
			cw .= c .* view(weights, j, :)
			cwc = dot(c, cw)
			iszero(cwc) ? eigvec[j, i] = 0 : eigvec[j, i] = dot(view(data, j, :), cw) / cwc
		end
		data .-= view(eigvec, :, i) * c'
	end
	#- Renormalize and re-orthogonalize the answer
	eigvec[:, 1] ./= norm(view(eigvec, :, 1))
	for k in 2:nvec
		for kx in 1:(k-1)
			c = dot(view(eigvec, :, k), view(eigvec, :, kx))
			eigvec[:, k] .-=  c .* view(eigvec, :, kx)
		end
		eigvec[:, k] ./= norm(view(eigvec, :, k))
	end
	# solve_model!(model, eigvec, coeff)
end

function _random_orthonormal(nvar, nvec)
	A = Array{Float64}(undef, nvar, nvec)
	keep_going = true
	i = 0
	while keep_going
		i += 1
		A .= randn(nvar, nvec)
		A[1, :] ./= norm(view(A, :, 1))
		for i in 2:nvec
			for j in 1:i
				A[:, i] .-= dot(view(A, :, j), view(A, :, i)) .* view(A, :, j)
				A[:, i] ./= norm(view(A, :, i))
			end
		end
		keep_going = any(isnan.(A)) && (i < 100)
	end
	if i > 99; println("_random_orthonormal() in empca failed for some reason") end
	return A
end

function _solve(
    dm::AbstractMatrix{T},
    data::AbstractVector,
    w::AbstractVector) where {T<:Real}
    return (dm' * (w .* dm)) \ (dm' * (w .* data))
end

function _empca!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; niter::Int=100, nvec::Int=5)

    #- Basic dimensions
    nvar, nobs = size(data)
    @assert size(data) == size(weights)
	@assert size(coeff, 1) == size(eigvec, 2) == nvec
	@assert size(coeff, 2) == nobs
	@assert size(eigvec, 1) == nvar

    #- Starting random guess
    eigvec .= _random_orthonormal(nvar, nvec)

	_solve_coeffs!(eigvec, coeff, data, weights)
	_data = copy(data)
    for k in 1:niter
		_solve_eigenvectors!(eigvec, coeff, data, weights)
		_data .= data
        _solve_coeffs!(eigvec, coeff, data, weights)
	end

    return eigvec, coeff
end
