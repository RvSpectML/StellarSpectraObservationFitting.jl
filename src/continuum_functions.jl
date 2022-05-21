# Heavily inspired (if not blatantly ripped of from) the functions at
# https://github.com/megbedell/wobble/blob/master/wobble/data.py
using LinearAlgebra

# using Plots
# These were initially defined to act on all of the orders of the spectra at a
# given time, but I have defined them to act on all of the times of the spectra
# at a given order. Should be equivalent
function fit_continuum(x::AbstractVector, y::AbstractVector, σ²::AbstractVector; order::Int=6, nsigma::Vector{<:Real}=[0.3,3.0], maxniter::Int=50, plot_stuff::Bool=false, edge_mask::Int=0)
    """Fit the continuum using sigma clipping
    Args:
        x: The wavelengths
        y: The log-fluxes
        σ² : variances for `ys`.
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: vector (low, high)
        maxniter: The maximum number of iterations to do
    Returns:
        The value of the continuum at the wavelengths in x
    """
    @assert 0 <= order < length(x)
    @assert length(x) == length(y) == length(σ²)
    @assert length(nsigma) == 2

    A = vander(x .- mean(x), order)
    m = fill(true, length(x))
	m[σ² .== Inf] .= false  # mask out the bad pixels
	if edge_mask > 0
		# m[edge_pad+1:edge_mask+edge_pad] .= false
		# m[end-edge_mask-edge_pad+1:end-edge_pad] .= false
		hold_left = y[1:edge_mask]
		hold_right = y[end-edge_mask+1:end]
		y[1:edge_mask] .= 1
		y[end-edge_mask+1:end] .= 1
	end
    μ = ones(length(x))
    for i in 1:maxniter
        w = general_lst_sq(view(A, m, :), view(y, m), view(σ², m))
        μ[:] = A * w
		# if plot_stuff
		# 	plt = scatter(x[m], y[m]; label="used")
		# 	scatter!(plt, x[.!m], y[.!m]; label="masked")
		# 	plot!(plt, x, μ; label="model")
		# 	display(plt)
		# end
        resid = y - μ
        # sigma = median(abs.(resid))
		sigma = std(resid)
        m_new = (-nsigma[1]*sigma) .< resid .< (nsigma[2]*sigma)
        if sum(m) == sum(m_new); break end
        m = m_new
    end
	if edge_mask > 0
		y[1:edge_mask] .= hold_left
		y[end-edge_mask+1:end] .= hold_right
	end
    return μ
end
function continuum_normalize!(d; kwargs...)
	continuum = ones(size(d.log_λ_obs, 1))
	for i in 1:size(d.log_λ_obs, 2)
		continuum[:] = fit_continuum(view(d.log_λ_obs, :, i), view(d.flux, :, i), view(d.var, :, i); kwargs...)
		d.flux[:, i] ./= continuum
		d.var[:, i] ./= continuum .* continuum
	end
end

function mask_low_pixels!(y::AbstractVector, σ²::AbstractVector; min_flux::Real= 0., padding::Int= 2)
	bad = y .< min_flux
	for i in eachindex(bad)
		bad[i] = bad[i] || !isfinite(y[i])
	end
	y[bad] .= min_flux
	l = length(bad)
	for i in findall(bad)
		bad[max(1, i - padding):min(i - padding, l)] .= true
	end
	σ²[bad] .= Inf
end
function mask_low_pixels!(d; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		mask_low_pixels!(view(d.flux, :, i), view(d.var, :, i); kwargs...)
	end
end

function mask_high_pixels!(y::AbstractVector, σ²::AbstractVector; max_flux::Real= 2., padding::Int= 2)
	bad = y .> max_flux
	for i in eachindex(bad)
		bad[i] = bad[i] || !isfinite(y[i])
	end
	y[bad] .= max_flux
	l = length(bad)
	for i in findall(bad)
		bad[max(1, i - padding):min(i - padding, l)] .= true
	end
	σ²[bad] .= Inf
end
function mask_high_pixels!(d; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		mask_high_pixels!(view(d.flux, :, i), view(d.var, :, i); kwargs...)
	end
end

function mask_bad_edges!(y::AbstractVector, σ²::AbstractVector; window_width::Int=128, min_snr::Real=5.)
	n_pix = length(y)
	for window_start in 1:Int(floor(window_width/10)):(n_pix - window_width)
		window_end = window_start + window_width
		mean_snr = sqrt(mean((y[window_start:window_end] .^2) ./ abs.(σ²[window_start:window_end])))
		if mean_snr > min_snr
			σ²[1:window_start] .= Inf # trim everything to left of window
			break
		end
	end
	for window_end in n_pix:-Int(floor(window_width/10)):(window_width + 1)
		window_start = window_end - window_width
		mean_snr = sqrt(mean((y[window_start:window_end] .^2) ./ abs.(σ²[window_start:window_end])))
		if mean_snr > min_snr
			σ²[window_end:end] .= Inf # trim everything to right of window
			break
		end
	end
end
function mask_bad_edges!(d; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		mask_bad_edges!(view(d.flux, :, i), view(d.var, :, i); kwargs...)
	end
end

function median_normalize!(d; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		continuum = median(view(d.flux, :, i)[.!(isnan.(view(d.flux, :, i)))])
		d.flux[:, i] ./= continuum
		d.var[:, i] ./= continuum * continuum
	end
end

function process!(d; kwargs...)
	median_normalize!(d)
	mask_low_pixels!(d)
	mask_bad_edges!(d)
	red_enough = minimum(d.log_λ_obs) > log(4410)  # is there likely to even be a continuum
	enough_points = (sum(isinf.(d.var)) / length(d.var)) < 0.5
	if (red_enough && enough_points); continuum_normalize!(d; kwargs...) end
	mask_high_pixels!(d)
end
