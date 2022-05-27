# functions related to calculating the PCA scores of time series spectra
using LinearAlgebra
using EMPCA

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


"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute the PCA component with the largest eigenvalue
X is data, r is vector of random numbers, s is preallocated memory
r && s  are of same length as each data point
"""
function compute_pca_component_RVSKL!(X::Matrix{T}, r::AbstractArray{T, 1}, s::Vector{T}; tol::Float64=1e-12, max_it::Int64=20) where {T<:Real}
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    @assert length(r) == num_lambda
    #rand!(r)  # assume r is already randomized
    last_mag_s = 0.0
    for j in 1:max_it
		s[:] = zeros(T, num_lambda)
		for i in 1:num_spectra
			BLAS.axpy!(dot(view(X, :, i), r), view(X, :, i), s)  # s += dot(X[:,i],r)*X[:,i]
		end
		mag_s = norm(s)
		r[:]  = s / mag_s
		if abs(mag_s - last_mag_s) < (tol * mag_s); break end
		last_mag_s = mag_s
	end
	return r
end


function project_doppler_comp!(M::AbstractMatrix, Xtmp::AbstractMatrix, scores::AbstractMatrix, fixed_comp::AbstractVector)
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

function find_PCA_comps!(M::AbstractMatrix, Xtmp::AbstractMatrix, scores::AbstractMatrix, s::AbstractVector; inds::UnitRange{<:Int}=1:size(M, 2), kwargs...)
	println(inds)
	@assert inds[1] > 0
	# remaining component calculations
	for j in inds
		compute_pca_component_RVSKL!(Xtmp, view(M, :, j), s; kwargs...)
		for i in 1:size(Xtmp, 2)
			scores[j, i] = dot(view(Xtmp, :, i), view(M, :, j)) #/sum(abs2,view(M,:,j-1))
			Xtmp[:,i] .-= scores[j, i] * view(M, :, j)
		end
	end
end

"""
modified code shamelessly stolen from RvSpectraKitLearn.jl/src/generalized_pca.jl
Compute first num_components basis vectors for PCA, after subtracting projection onto fixed_comp
"""
function fit_gen_pca_rv_RVSKL(X::Matrix{T}, fixed_comp::Vector{T}; mu::Vector{T}=vec(mean(X, dims=2)), num_components::Integer=3, kwargs...) where {T<:Real}

	# initializing relevant quantities
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    M = rand(T, (num_lambda, num_components))  # random initialization is part of algorithm (i.e., not zeros)
    s = zeros(T, num_lambda)  # pre-allocated memory for compute_pca_component
    scores = zeros(num_components, num_spectra)

    Xtmp = X .- mu  # perform PCA after subtracting off mean

	# doppler component calculations
	rvs = project_doppler_comp!(M, Xtmp, scores, fixed_comp)

	if num_components>1
		find_PCA_comps!(M, Xtmp, scores, s; inds=2:num_components)
	end

	return (mu, M, scores, rvs)
end


function fit_gen_pca(X::Matrix{T}; mu::Vector{T}=vec(mean(X, dims=2)), num_components::Integer=2) where {T<:Real}

	# initializing relevant quantities
	num_lambda = size(X, 1)
    num_spectra = size(X, 2)
    M = rand(T, (num_lambda, num_components))  # random initialization is part of algorithm (i.e., not zeros)
    s = zeros(T, num_lambda)  # pre-allocated memory for compute_pca_component
    scores = zeros(num_components, num_spectra)

    Xtmp = X .- mu  # perform PCA after subtracting off mean

	find_PCA_comps!(M, Xtmp, scores, s)

	return (mu, M, scores)
end

function DPCA(spectra::Matrix{T}, λs::Vector{T};
	template::Vector{T}=make_template(spectra), kwargs...) where {T<:Real}

	doppler_comp = calc_doppler_component_RVSKL(λs, template)
    return fit_gen_pca_rv_RVSKL(spectra, doppler_comp; mu=template, kwargs...)
end


function EMPCA!(M::AbstractMatrix, Xtmp::AbstractMatrix, scores::AbstractMatrix, weights::AbstractMatrix; inds::UnitRange{<:Int}=1:size(M, 2), kwargs...)
	@assert inds[1] > 0
	m = empca.empca(Xtmp', weights', nvec=length(inds), silent=true, kwargs...)
	M[:, inds] .= m.eigvec'
	scores[inds, :] .= m.coeff'
end

function DEMPCA!(spectra::Matrix{T}, λs::Vector{T}, M::AbstractMatrix, scores::AbstractMatrix, weights::Matrix{T};
	template::Vector{T}=make_template(spectra), kwargs...) where {T<:Real}
	doppler_comp = calc_doppler_component_RVSKL(λs, template)
    return DEMPCA!(M, spectra .- template, scores, weights, doppler_comp; kwargs...)
end

function DEMPCA!(M::AbstractMatrix, Xtmp::AbstractMatrix, scores::AbstractMatrix, weights::AbstractMatrix, doppler_comp::Vector{T}; kwargs...) where {T<:Real}
	rvs = project_doppler_comp!(M, Xtmp, scores, doppler_comp)
	if size(M, 2) > 1
		EMPCA!(M, Xtmp, scores, weights; inds=2:size(M, 2), kwargs...)
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
