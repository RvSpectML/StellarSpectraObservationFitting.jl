using UnitfulAstro, Unitful
using Statistics
using LinearAlgebra

light_speed = uconvert(u"m/s", 1u"c")
light_speed_nu = ustrip(light_speed)

function searchsortednearest(a::AbstractVector{T} where T<:Real, x::Real; lower::Bool=false)
   	idx = searchsortedfirst(a,x)
   	if (idx==1); return idx; end
   	if (idx>length(a)); return length(a); end
   	if (a[idx]==x); return idx; end
   	if lower || ((abs(a[idx]-x) >= abs(a[idx-1]-x)))
      	return idx - 1
   	else
      	return idx
   	end
end


function searchsortednearest(a::AbstractVector{T} where T<:Real, x::AbstractVector{T} where T<:Real; kwargs...)
	@assert issorted(x)
	len_x = length(x)
   	len_a = length(a)
   	idxs = Array{Int64}(undef, len_x)
   	idxs[1] = searchsortednearest(a, x[1]; kwargs...)
	for i in 2:len_x
	   	idxs[i] = idxs[i-1] + searchsortednearest(view(a, idxs[i-1]:len_a), x[i]; kwargs...) - 1
   	end
   	return idxs
end

function clip_vector!(vec::Vector; max::Number=Inf, min::Number=-Inf)
	vec[vec .< min] .= min
	vec[vec .> max] .= max
end

function make_template(matrix::Matrix; use_mean::Bool=false, kwargs...)
	if use_mean
		result = vec(mean(matrix, dims=2))
	else
		result = vec(median(matrix, dims=2))
	end
	clip_vector!(result; kwargs...)
	return result
end

function shift_log_λ(v::Unitful.Velocity, log_λ::Vector{T}) where {T<:Real}
	return log_λ .+ (log((1.0 + v / light_speed) / (1.0 - v / light_speed)) / 2)
end

function observation_night_inds(times_in_days::Vector{<:Real})
    difs = times_in_days[2:end] - times_in_days[1:end-1] .> 0.5
    obs_in_first_night = findfirst(difs)
    if isnothing(findfirst(difs))
        return [1:length(times_in_days)]
    else
        night_inds = [1:findfirst(difs)]
    end
    i = night_inds[1][end]
    while i < length(times_in_days)
        obs_in_night = findfirst(view(difs, i:length(difs)))
        if isnothing(obs_in_night)
            i = length(times_in_days)
        else
            i += obs_in_night
            append!(night_inds, [night_inds[end][end]+1:i])
        end
    end
    return night_inds
end
observation_night_inds(times::Vector{<:Unitful.Time}) =
    observation_night_inds(ustrip.(uconvert.(u"d", times)))

function copy_dict!(from::Dict, to::Dict)
    for (key, value) in from
		to[key] = from[key]
	end
end

function parse_args(ind::Int, type::DataType, default)
	@assert typeof(default) <: type
    if length(ARGS) > (ind - 1)
        return parse(type, ARGS[ind])
    else
        return default
    end
end


function banded_inds(row::Int, span::Int, row_len::Int)
	low = max(row - span, 1); high = min(row + span, row_len);
	return low, high
end

function vander(x::AbstractVector, n::Int)
    m = ones(length(x), n + 1)
    for i in 1:n
        m[:, i + 1] .= m[:, i] .* x
    end
    return m
end

"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function general_lst_sq(
    dm::AbstractMatrix{T},
    data::AbstractVector,
    Σ::Union{Cholesky,Diagonal}) where {T<:Real}
    return (dm' * (Σ \ dm)) \ (dm' * (Σ \ data))
end
general_lst_sq(dm, data, σ²::AbstractVector) =
	general_lst_sq(dm, data, Diagonal(σ²))

"""
Solve a linear system of equations (optionally with variance values at each point or covariance array)
see (https://en.wikipedia.org/wiki/Generalized_least_squares#Method_outline)
"""
function ordinary_lst_sq(
    dm::AbstractMatrix{T},
    data::AbstractVector) where {T<:Real}
    return (dm' * dm) \ (dm' * data)
end
general_lst_sq(dm, data) = ordinary_lst_sq(dm, data)

"a generalized version of the built in append!() function"
function multiple_append!(a::Vector{T}, b...) where {T<:Real}
    for i in eachindex(b)
        append!(a, b[i])
    end
    return a
end

const _fwhm_2_σ_factor = 1 / (2 * sqrt(2 * log(2)))
fwhm_2_σ(fwhm::Real) = _fwhm_2_σ_factor * fwhm

ordinary_lst_sq(
    data::AbstractVector,
    order::Int;
	x::AbstractVector=1:length(data)) = ordinary_lst_sq(vander(x, order), data)

function ordinary_lst_sq_f(data::AbstractVector, order::Int; kwargs...)
	w = ordinary_lst_sq(data, order; kwargs...)
	return x -> LinearAlgebra.BLAS.dot([x ^ i for i in 0:order], w)
	# faster than the following
	# return x -> ([x ^ i for i in 0:order]' * w)
	# return x -> mapreduce(i -> w[i+1] * x ^ i , +, 0:order)
	# return x -> sum([x ^ i for i in 0:order] .* w)
end

_trapzx2(x1::Real, x2::Real, y1::Real, y2::Real) = (x2 - x1) * (y1 + y2)
# _trapz_large(x::AbstractVector, y::AbstractVector) =
# 	mapreduce(i -> (x[i+1] - x[i]) * (y[i] + y[i+1]), +, 1:(length(y) - 1)) / 2
# function trapz_large(x::AbstractVector, y::AbstractVector)
#     @assert length(x) == length(y) > 0 "x and y vectors must be of the same (non-zero) length!"
# 	return _trapz_large(x, y)
# end
# trapz(x::AbstractVector, y::AbstractVector) = trapz_large(x, y)

"""
trapezoidal integration, shamelessly modified from. Faster for small ranges
https://github.com/dextorious/NumericalIntegration.jl/blob/master/src/NumericalIntegration.jl
"""
function trapz_small(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "x and y vectors must be of the same length!"
    integral =  0
    @fastmath @simd for i in 1:(length(y) - 1)
    # @simd for i in 1:(length(y) - 1)
        @inbounds integral += _trapzx2(x[i], x[i+1], y[i], y[i+1])
    end
    return integral / 2
end

function trapz_small(lo_x::Real, hi_x::Real, x::AbstractVector, y::AbstractVector)
    lo_ind, hi_ind = searchsortednearest(x, [lo_x, hi_x])
    # make sure that the inds are inside lo_x and hi_x
    if x[lo_ind] < lo_x; lo_ind += 1 end
    if x[hi_ind] > hi_x; hi_ind -= 1 end
    # integrate over main section + edges
	integral = trapz_small(view(x, lo_ind:hi_ind), view(y, lo_ind:hi_ind)) +
		+ _trapzx2(lo_x, x[lo_ind], y[lo_ind-1] + ((lo_x - x[lo_ind-1]) / (x[lo_ind]-x[lo_ind-1]) * (y[lo_ind] - y[lo_ind-1])), y[lo_ind])
		+ _trapzx2(x[hi_ind], hi_x, y[hi_ind], y[hi_ind] + ((hi_x - x[hi_ind]) / (x[hi_ind+1]-x[hi_ind]) * (y[hi_ind+1] - y[hi_ind])))
    return integral
end
# using Test
# @testset "trapz tests" begin
#     xs = ys = collect(1.:20)
#     integral(x1, x2) = (x2^2 - x1^2) / 2
#     for x1 in [1.9, 2., 2.1]
#         for x2 in [10.9, 11., 11.1]
#             @test isapprox(trapz(x1, x2, xs, ys), integral(x1, x2))
#         end
#     end
# end


oversamp_interp(lo_x::Real, hi_x::Real, x::AbstractVector, y::AbstractVector) =
	trapz_small(lo_x, hi_x, x, y) / (hi_x - lo_x)
# function undersamp_interp(x_new::Real, x::AbstractVector, y::AbstractVector)
# 	ind = searchsortednearest(x, x_new; lower=true)
# 	dif = (x_new-x[ind]) / (x[ind+1] - x[ind])
# 	return y[ind] * (1-dif) + y[ind+1] * dif
# end

# pixel_separation(xs::AbstractVector) = multiple_append!([xs[1] - xs[2]], (xs[1:end-2] - xs[3:end]) ./ 2, [xs[end-1] - xs[end]])
function bounds_generator!(bounds::AbstractVector, xs::AbstractVector)
	bounds[1] = (3*xs[1] - xs[2]) / 2
	bounds[2:end-1] = (view(xs, 1:length(xs)-1) .+ view(xs, 2:length(xs))) ./ 2
	bounds[end] = (3*xs[end] - xs[end-1]) / 2
	return bounds
end
function bounds_generator(xs::AbstractVector)
	bounds = Array{Float64}(undef, length(xs)+1)
	bounds_generator!(bounds, xs)
	return bounds
end
function bounds_generator(xs::AbstractMatrix)
	bounds = Array{Float64}(undef, size(xs, 1)+1, size(xs, 2))
	for i in 1:size(xs, 2)
		bounds_generator!(view(bounds, :, i), view(xs, :, i))
	end
	return bounds
end

# converts wavelength (Å) to wavenumber (1/cm) (and vice versa)
Å_to_wavenumber(λ::Real) = 1e8 / λ
wavenumber_to_Å(wn::Real) = Å_to_wavenumber(wn)

vector_zero(θ::VecOrMat) = zero(θ)
vector_zero(θ::Vector{<:Array}) = [vector_zero(i) for i in θ]
