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
   	len_x = length(x)
   	len_a = length(a)
   	idxs = zeros(Int64, len_x)
   	idxs[1] = searchsortednearest(a, x[1]; lower=lower)
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
        m[:, i + 1] = m[:, i] .* x
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
    for i in 1:length(b)
        append!(a, b[i])
    end
    return a
end

const _fwhm_2_σ_factor = 1 / (2 * sqrt(2 * log(2)))
fwhm_2_σ(fwhm) = _fwhm_2_σ_factor .* fwhm
