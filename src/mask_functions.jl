function mask_pixels!(d::Data, inds::AbstractVector; print_something::Bool=true)
	d.var[inds, :] .= Inf
	if print_something; println("masked some pixels") end
end

function mask_tellurics!(d::Data, log_λ_low::Real, log_λ_high::Real; print_something::Bool=true)
	@assert log_λ_low < log_λ_high
	d.var[log_λ_low .< d.log_λ_obs .< log_λ_high] .= Inf
	if print_something; println("masked some pixels in the telluric frame") end
end

function mask_stellar_features!(d::Data, log_λ_low::Real, log_λ_high::Real; print_something::Bool=true)
	@assert log_λ_low < log_λ_high
	d.var[log_λ_low .< d.log_λ_star .< log_λ_high] .= Inf
	if print_something; println("masked some pixels in the stellar frame") end
end
