using UnitfulAstro, Unitful
using Statistics

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


function searchsortednearest(a::AbstractVector{T} where T<:Real, x::AbstractVector{T} where T<:Real; lower::Bool=false)
   	len_x = length(x)
   	len_a = length(a)
   	idxs = zeros(Int64, len_x)
   	idxs[1] = searchsortednearest(a, x[1]; lower=lower)
	for i in 2:len_x
	   	idxs[i] = idxs[i-1] + searchsortednearest(view(a, idxs[i-1]:len_a), x[i]; lower=lower) - 1
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
