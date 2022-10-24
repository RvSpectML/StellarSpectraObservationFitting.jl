## EMPCA implementation
# https://github.com/sbailey/empca
# S. Bailey 2012, PASP, 124, 1015

using LinearAlgebra

function EMPCA!(M::AbstractMatrix, scores::AbstractMatrix, μ::AbstractVector, Xtmp::AbstractMatrix, weights::AbstractMatrix; log_lm::Bool=false, kwargs...)
	if log_lm
		weights .*= (Xtmp .^ 2)
		mask = weights.!=0
		Xtmp[mask] = log.(view(Xtmp ./ μ, mask))
		Xtmp[.!mask] .= 0
	else
		Xtmp .-= μ
	end
	_empca!(M, scores, Xtmp, weights; kwargs...)
end

function _empca!(M::AbstractMatrix, scores::AbstractMatrix, Xtmp::AbstractMatrix, weights::AbstractMatrix; inds::UnitRange{<:Int}=1:size(M, 2), vec_by_vec::Bool=true, kwargs...)
	if length(inds) > 0
		@assert inds[1] > 0
		vec_by_vec ?
			_empca_vec_by_vec!(M, scores, Xtmp, weights; inds=inds, kwargs...) :
			_empca_all_at_once!(M, scores, Xtmp, weights; inds=inds, kwargs...)
	end
end

function _solve_coeffs!(eigvec::AbstractVector, coeff::AbstractVector, data::AbstractMatrix, weights::AbstractMatrix)
	for i in 1:size(data, 2)
		coeff[i] = _solve(eigvec, view(data, :, i), view(weights, :, i))
	end
end
function _solve_coeffs!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; inds::UnitRange{<:Int}=1:size(eigvec, 2))
	for i in 1:size(data, 2)
		coeff[inds, i] .= _solve(view(eigvec, :, inds), view(data, :, i), view(weights, :, i))
	end
	# solve_model!(model, eigvec, coeff)
end

function _solve_eigenvectors!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; inds::UnitRange{<:Int}=1:size(eigvec, 2))
	nvar = size(eigvec, 1)
	cw = Array{Float64}(undef, size(data, 2))
	for i in inds
		c = view(coeff, i, :)
		for j in 1:nvar
			cw[:] = c .* view(weights, j, :)
			cwc = dot(c, cw)
			iszero(cwc) ? eigvec[j, i] = 0 : eigvec[j, i] = dot(view(data, j, :), cw) / cwc
		end
		data .-= view(eigvec, :, i) * c'
	end
	eigvec[:, 1] ./= norm(view(eigvec, :, 1))
	_reorthogonalize(eigvec)
	# solve_model!(model, eigvec, coeff)
end
function _solve_eigenvectors!(eigvec::AbstractVector, coeff::AbstractVector, data::AbstractMatrix, weights::AbstractMatrix)
	nvar = length(eigvec)
	cw = Array{Float64}(undef, size(data, 2))
	for j in 1:nvar
		cw[:] = coeff .* view(weights, j, :)
		cwc = dot(coeff, cw)
		iszero(cwc) ? eigvec[j] = 0 : eigvec[j] = dot(view(data, j, :), cw) / cwc
	end
	# Renormalize the answer
	eigvec ./= norm(eigvec)
end
function _reorthogonalize!(eigvec::AbstractMatrix)
	#- Renormalize and re-orthogonalize the answer
	nvec = size(eigvec, 2)
	if nvec > 1
		for k in 2:nvec
			for kx in 1:(k-1)
				c = dot(view(eigvec, :, k), view(eigvec, :, kx))
				eigvec[:, k] .-=  c .* view(eigvec, :, kx) / sum(abs2, view(eigvec, :, kx))
			end
			eigvec[:, k] ./= norm(view(eigvec, :, k))
		end
	end
end

function _random_orthonormal!(A::AbstractMatrix, nvar::Int; log_λ::Union{Nothing, AbstractVector}=nothing, inds::UnitRange{<:Int}=1:size(A, 2))
	keep_going = true
	i = 0
	while keep_going
		i += 1
		if log_λ != nothing
			fx = SOAP_gp(log_λ, SOAP_gp_var)
			for i in inds
				A[:, i] = rand(fx)
			end
		else
			A[:, inds] .= randn(nvar, length(inds))
		end
		for i in inds
			for j in 1:(i-1)
				A[:, i] .-= dot(view(A, :, j), view(A, :, i)) .* view(A, :, j)
			end
			A[:, i] ./= norm(view(A, :, i))
		end
		keep_going = any(isnan.(A)) && (i < 100)
	end
	if i > 99; println("_random_orthonormal!() in empca failed for some reason") end
	return A
end

function _solve(
    dm::AbstractVecOrMat{T},
    data::AbstractVector,
    w::AbstractVector) where {T<:Real}
    return (dm' * (w .* dm)) \ (dm' * (w .* data))
end

function _empca_all_at_once!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; niter::Int=100, inds::UnitRange{<:Int}=1:size(eigvec, 2), kwargs...)

    #- Basic dimensions
    nvar, nobs = size(data)
    @assert size(data) == size(weights)
	@assert size(coeff, 1) == size(eigvec, 2)
	@assert size(coeff, 2) == nobs
	@assert size(eigvec, 1) == nvar

    #- Starting random guess
    eigvec .= _random_orthonormal!(eigvec, nvar; inds=inds, kwargs...)

	_solve_coeffs!(eigvec, coeff, data, weights)
	_data = copy(data)
    for k in 1:niter
		_solve_eigenvectors!(eigvec, coeff, _data, weights; inds=inds)
		_data .= data
        _solve_coeffs!(eigvec, coeff, _data, weights; inds=inds)
	end

    return eigvec, coeff
end


function _empca_vec_by_vec!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; niter::Int=100, inds::UnitRange{<:Int}=1:size(eigvec, 2), kwargs...)

    #- Basic dimensions
    nvar, nobs = size(data)
    @assert size(data) == size(weights)
	@assert size(coeff, 1) == size(eigvec, 2)
	@assert size(coeff, 2) == nobs
	@assert size(eigvec, 1) == nvar

	_data = copy(data)
	for i in inds
		eigvec[:, i] .= randn(nvar)
		# eigvec[:, i] ./= norm(view(eigvec, :, i))
		_reorthogonalize!(view(eigvec, :, 1:i))  # actually useful
		_solve_coeffs!(view(eigvec, :, i), view(coeff, i, :), data, weights)
	    for k in 1:niter
			_solve_eigenvectors!(view(eigvec, :, i), view(coeff, i, :), _data, weights)
			_reorthogonalize!(view(eigvec, :, 1:i))  # actually useful
	        _solve_coeffs!(view(eigvec, :, i), view(coeff, i, :), _data, weights)
		end
		_data .-= view(eigvec, :, i) * view(coeff, i, :)'
	end

    return eigvec, coeff
end
