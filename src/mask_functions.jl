insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), [x]); v)
function affected_pixels(bad::AbstractVecOrMat)
	affected = Int64[]
	for ij in findall(bad)
		insert_and_dedup!(affected, ij[1])
	end
	return affected
end
function affected_pixels!(affected1::Vector, affected2::AbstractVector)
	for i in affected2
		insert_and_dedup!(affected1, i)
	end
	return affected1
end

function _mask(var::AbstractVecOrMat, bad_inds; using_weights::Bool=false)
	if using_weights
		var[bad_inds] .= 0
	else
		var[bad_inds] .= Inf
	end
	return affected_pixels(bad_inds)
end

# function mask_pixels!(var::AbstractMatrix, inds::AbstractVector; print_something::Bool=true)
# 	var[inds, :] .= Inf
# 	if print_something; println("masked some pixels") end
# end
# mask_pixels!(d::Data, inds::AbstractVector; kwargs...) = mask_pixels!(d.var, inds; kwargs...)

function mask_stellar_feature!(var::AbstractMatrix, log_λ_star::AbstractMatrix, log_λ_low::Real, log_λ_high::Real; print_something::Bool=true, inverse::Bool=false, kwargs...)
	@assert log_λ_low < log_λ_high
	inverse ?
		bad = .!(log_λ_low .< log_λ_star .< log_λ_high) :
		bad = log_λ_low .< log_λ_star .< log_λ_high
	_mask(var, bad; kwargs...)
	if print_something; println("masked some features in the stellar frame") end
	return affected_pixels(bad)
end
mask_stellar_feature!(d::Data, log_λ_low::Real, log_λ_high::Real; kwargs...) = mask_stellar_feature!(d.var, d.log_λ_star, log_λ_low, log_λ_high; kwargs...)

function mask_telluric_feature!(var::AbstractMatrix, log_λ_obs::AbstractMatrix, log_λ_star::AbstractMatrix, log_λ_low::Real, log_λ_high::Real; print_something::Bool=true, in_stellar_frame::Bool=true, kwargs...)
	@assert log_λ_low < log_λ_high
	if in_stellar_frame
		log_λ_low_star, log_λ_high_star = extrema(log_λ_star[log_λ_low .< log_λ_obs .< log_λ_high])
		if print_something; println("masked some telluric features in the stellar frame") end
		return mask_stellar_feature!(var, log_λ_star, log_λ_low_star, log_λ_high_star; print_something=false, kwargs...)
	else
		if print_something; println("masked some features in the telluric frame") end
		return _mask(var, log_λ_low .< log_λ_obs .< log_λ_high; kwargs...)
	end
end
mask_telluric_feature!(d::Data, log_λ_low::Real, log_λ_high::Real; kwargs...) = mask_telluric_feature!(d.var, d.log_λ_obs, d.log_λ_star, log_λ_low, log_λ_high; kwargs...)

function mask_stellar_pixel!(var::AbstractMatrix, log_λ_star::AbstractMatrix, log_λ_star_bounds::AbstractMatrix, i::Int; padding::Int=0, print_something::Bool=true, kwargs...)
	log_λ_low = minimum(view(log_λ_star_bounds, max(1, i - padding), :))
	log_λ_high = maximum(view(log_λ_star_bounds, min(i + padding, size(log_λ_star_bounds, 1)), :))
	if print_something; println("masked some pixels in the stellar frame") end
	return mask_stellar_feature!(var, log_λ_star, log_λ_low, log_λ_high; print_something=false, kwargs...)
end
function mask_stellar_pixel!(var::AbstractMatrix, log_λ_star::AbstractMatrix, log_λ_star_bounds::AbstractMatrix, inds::AbstractVector; print_something::Bool=true, kwargs...)
	affected = Int64[]
		if length(inds) > 0
		if print_something; println("masked some pixels in the stellar frame") end
		for i in inds
			affected2 = mask_stellar_pixel!(var, log_λ_star, log_λ_star_bounds, i; print_something=false, kwargs...)
			affected_pixels!(affected, affected2)
		end
	end
	return affected
end
mask_stellar_pixel!(d::Data, inds_or_i; kwargs...) = mask_stellar_pixel!(d.var, d.log_λ_star, d.log_λ_star_bounds, inds; kwargs...)
