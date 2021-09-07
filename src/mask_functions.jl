function mask_pixels!(d::Data, inds::AbstractVector)
	d.var[inds, :] .= Inf
end

function mask_tellurics!(d::Data, log_λ_low::Real, log_λ_high::Real)
	@assert log_λ_low < log_λ_high
	d.var[log_λ_low .< d.log_λ_obs .< log_λ_high] .= Inf
end

function mask_stellar_features!(d::Data, log_λ_low::Real, log_λ_high::Real)
	@assert log_λ_low < log_λ_high
	d.var[log_λ_low .< d.log_λ_star .< log_λ_high] .= Inf
end
