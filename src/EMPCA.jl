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
	@assert inds[1] > 0
	vec_by_vec ?
		_empca_vec_by_vec!(view(M, :, inds), view(scores, inds, :), Xtmp, weights; nvec=length(inds), kwargs...) :
		_empca_all_at_once!(view(M, :, inds), view(scores, inds, :), Xtmp, weights; nvec=length(inds), kwargs...)
end

function _solve_coeffs!(eigvec::AbstractVector, coeff::AbstractVector, data::AbstractMatrix, weights::AbstractMatrix)
	for i in 1:size(data, 2)
		coeff[i] = _solve(eigvec, view(data, :, i), view(weights, :, i))
	end
end
function _solve_coeffs!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix)
	for i in 1:size(data, 2)
		coeff[:, i] .= _solve(eigvec, view(data, :, i), view(weights, :, i))
	end
	# solve_model!(model, eigvec, coeff)
end

function _solve_eigenvectors!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix)
	nvar, nvec = size(eigvec)
	cw = Array{Float64}(undef, size(data, 2))
	for i in 1:nvec
		c = view(coeff, i, :)
		for j in 1:nvar
			cw[:] = c .* view(weights, j, :)
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

function _random_orthonormal(nvar::Int, nvec::Int; log_λ::Union{Nothing, AbstractVector}=nothing)
	A = Array{Float64}(undef, nvar, nvec)
	keep_going = true
	i = 0
	while keep_going
		i += 1
		if log_λ != nothing
			fx = SOAP_gp(log_λ, SOAP_gp_var)
			for i in 1:nvec
				A[:, i] = rand(fx)
			end
		else
			A .= randn(nvar, nvec)
		end
		A[:, 1] ./= norm(view(A, :, 1))
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
    dm::AbstractVecOrMat{T},
    data::AbstractVector,
    w::AbstractVector) where {T<:Real}
    return (dm' * (w .* dm)) \ (dm' * (w .* data))
end

function _empca_all_at_once!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; niter::Int=100, nvec::Int=5, kwargs...)

    #- Basic dimensions
    nvar, nobs = size(data)
    @assert size(data) == size(weights)
	@assert size(coeff, 1) == size(eigvec, 2) == nvec
	@assert size(coeff, 2) == nobs
	@assert size(eigvec, 1) == nvar

    #- Starting random guess
    eigvec .= _random_orthonormal(nvar, nvec; kwargs...)

	_solve_coeffs!(eigvec, coeff, data, weights)
	_data = copy(data)
    for k in 1:niter
		_solve_eigenvectors!(eigvec, coeff, _data, weights)
		_data .= data
        _solve_coeffs!(eigvec, coeff, _data, weights)
	end

    return eigvec, coeff
end


function _empca_vec_by_vec!(eigvec::AbstractMatrix, coeff::AbstractMatrix, data::AbstractMatrix, weights::AbstractMatrix; niter::Int=100, nvec::Int=5, kwargs...)

    #- Basic dimensions
    nvar, nobs = size(data)
    @assert size(data) == size(weights)
	@assert size(coeff, 1) == size(eigvec, 2) == nvec
	@assert size(coeff, 2) == nobs
	@assert size(eigvec, 1) == nvar

	_data = copy(data)
	for i in 1:nvec
		# I don't believe I have to explicitly enforce orthagonality as it is
		# implicitly enforced by ftting the data without previous component's
		# variances
		eigvec[:, i] .= randn(nvar)
		eigvec[:, i] ./= norm(view(eigvec, :, i))

		_solve_coeffs!(view(eigvec, :, i), view(coeff, i, :), data, weights)
	    for k in 1:niter
			_solve_eigenvectors!(view(eigvec, :, i), view(coeff, i, :), _data, weights)
	        _solve_coeffs!(view(eigvec, :, i), view(coeff, i, :), _data, weights)
		end
		_data .-= view(eigvec, :, i) * view(coeff, i, :)'
	end

    return eigvec, coeff
end
