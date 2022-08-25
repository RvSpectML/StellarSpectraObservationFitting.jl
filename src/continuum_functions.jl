# Heavily inspired (if not blatantly ripped of from) the functions at
# https://github.com/megbedell/wobble/blob/master/wobble/data.py
using LinearAlgebra
using Statistics
import StatsBase: winsor

# using Plots

_high_quantile_default = 0.9
# These were initially defined to act on all of the orders of the spectra at a
# given time, but I have defined them to act on all of the times of the spectra
# at a given order. Should be equivalent
function fit_continuum(x::AbstractVector, y::AbstractVector, σ²::AbstractVector; ignore_weights::Bool=false, order::Int=6, nsigma::Vector{<:Real}=[0.3,3.0], maxniter::Int=50, plot_stuff::Bool=false, edge_mask::Int=0)
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
	# σ² = copy(σ²)
	# σ²_thres = quantile(σ²[.!isinf.(σ²)], _high_quantile_default)/10
	# σ²[σ² .< σ²_thres] .= σ²_thres
	m[y .< 0.5] .= false  # mask out the bad pixels
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
	w = Array{Float64}(undef, order+1)
    for i in 1:maxniter
		if ignore_weights
			w[:] = general_lst_sq(view(A, m, :), view(y, m))
        else
			w[:] = general_lst_sq(view(A, m, :), view(y, m), view(σ², m))
		end
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
		m_new[y .< 0.5] .= false
		m_new[σ² .== Inf] .= false  # mask out the bad pixels
        if sum(m) == sum(m_new); break end
        m = m_new
    end
	if edge_mask > 0
		y[1:edge_mask] .= hold_left
		y[end-edge_mask+1:end] .= hold_right
	end
    return μ, w
end
function continuum_normalize!(d::Data; order::Int=6, kwargs...)
	continuum = ones(size(d.log_λ_obs, 1))
	w = Array{Float64}(undef, order + 1, size(d.flux, 2))
	for i in 1:size(d.log_λ_obs, 2)
		continuum[:], w[:, i] = fit_continuum(view(d.log_λ_obs, :, i), view(d.flux, :, i), view(d.var, :, i); order=order, kwargs...)
		d.flux[:, i] ./= continuum
		d.var[:, i] ./= continuum .* continuum
	end
	return w
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
function mask_low_pixels!(d::Data; kwargs...)
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
function mask_high_pixels!(d::Data; kwargs...)
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
function mask_bad_edges!(d::Data; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		mask_bad_edges!(view(d.flux, :, i), view(d.var, :, i); kwargs...)
	end
end

function flat_normalize!(d::Data; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		continuum = quantile(view(d.flux, .!(isnan.(view(d.flux, :, i))), i), _high_quantile_default)
		d.flux[:, i] ./= continuum
		d.var[:, i] ./= continuum * continuum
	end
end

function outlier_mask(v::AbstractVecOrMat; thres::Real=10, prop::Real=0.2)
	wv = winsor(v; prop=prop)
	μ = mean(wv)
	σ = stdm(wv, μ)
	return (v .< (μ + thres * σ)) .&& (v .> (μ - thres * σ))
end

function recognize_bad_normalization!(d::Data; kwargs...)
	mask = outlier_mask([mean(view(d.var, .!isinf.(view(d.var, :, i)), i)) for i in 1:size(d.var, 2)]; kwargs...) .|| outlier_mask(vec(std(d.flux; dims=1)); kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		if !mask[i]
			# d.var[:, i] .= Inf
			println("spectrum $i has a weird continuum normalization, consider removing it from your analysis")
		end
	end
end

function recognize_bad_drift!(d::Data; kwargs...)
	mask = outlier_mask(d.log_λ_obs[1, :]; kwargs...)
	for i in 1:size(d.log_λ_obs, 2)
		if !mask[i]
			# d.var[:, i] .= Inf
			println("spectrum $i has a weird drift, consider removing it from your analysis")
		end
	end
end

function process!(d; λ_thres::Int=4000, kwargs...)
	flat_normalize!(d)
	mask_low_pixels!(d)
	mask_high_pixels!(d)
	mask_bad_edges!(d)
	# λ_thres = 4000 # is there likely to even be a continuum (neid index order 22+)
	# λ_thres = 6200 # where neid blaze correction starts to break down (neid index order 77+)
	# red_enough = minimum(d.log_λ_obs) > log(6200)
	red_enough = minimum(d.log_λ_obs) > log(λ_thres)
	enough_points = (sum(isinf.(d.var)) / length(d.var)) < 0.5
	if (red_enough && enough_points)
		w = continuum_normalize!(d; kwargs...)
	else
		w = nothing
	end
	recognize_bad_normalization!(d; thres=20)
	recognize_bad_drift!(d; thres=20)
	return w
end
