module NEIDLSF
    import StellarSpectraObservationFitting as SSOF
    using FITSIO
    using SpecialFunctions
    using SparseArrays
    # using Interpolations
    using DataInterpolations
    
    function conv_gauss_tophat(x::Real, σ::Real, boxhalfwidth::Real; amp::Real=1)
        scale = 1 / (sqrt(2) * σ)
        arg1 = (boxhalfwidth + x) * scale
        arg2 = (boxhalfwidth - x) * scale
        return abs(amp * (erf(arg1) + erf(arg2)))
        # return abs(amp * (erf(arg1) + erf(arg2)) / (2 * erf(boxhalfwidth * scale)))  # used for some visual diagnostic
    end

    # def conv_gauss_tophat(x, center, amp, sigma, boxhalfwidth):
    #     '''
    #     this is an analytical function for the convolution of a gaussian and tophat
    #     should be a closer approximation to the HPF profile than a simple gaussian
    #     '''
    #     arg1 = (2. * center + boxhalfwidth - 2. * x) / (2. * sqrt(2) * sigma)
    #     arg2 = (-2. * center + boxhalfwidth + 2. * x) / (2. * sqrt(2) * sigma)
    #     part1 = scipy.special.erf(arg1)
    #     part2 = scipy.special.erf(arg2)
    #     out = amp * (part1 + part2)
    #     return(out)

    σs = read(FITS(pkgdir(SSOF) *"/examples/data/sigma_arr.fits")[1])
    bhws = read(FITS(pkgdir(SSOF) *"/examples/data/boxhalfwidth_arr.fits")[1]) ./ 2 # Sam's formula has an extra factor of 2
    no_lsf_orders = [all(iszero.(view(bhws, :, i))) for i in axes(bhws,2)]
    @assert all(no_lsf_orders .== [all(iszero.(view(σs, :, i))) for i in axes(σs,2)])

    # function conv_gauss_tophat_integral(σ::Real, bhw::Real, xmμ::Real)
    #     x1 = xmμ - 0.5
    #     x2 = xmμ + 0.5
    #     scale = 1 / (sqrt(2) * σ)
    #     z1 = (bhw + x1) * scale
    #     zm1 = (bhw - x1) * scale
    #     z2 = (bhw + x2) * scale
    #     zm2 = (bhw - x2) * scale
    #     return sqrt(2 / π) * σ * (exp(-(zm1^2)) - exp(-(z1^2)) - exp(-(zm2^2)) + exp(-(z2^2))) +
    #         (bhw - x1) * erf(zm1) - (bhw + x1) * erf(z1) - (bhw - x2) * erf(zm2) + (bhw + x2) * erf(z2)
    # end

    threeish_sigma(σ::Real, bhw::Real) = 3 * abs(σ) + 0.87 * abs(bhw)
    function neid_lsf(order::Int)
        @assert 1 <= order <= length(no_lsf_orders)
        if no_lsf_orders[order]; return nothing end
        n = size(σs, 1)
        holder = zeros(n, n)
        for i in 1:n
            lo = max(1, Int(round(i - threeish_sigma(σs[i, order], bhws[i, order]))))
            hi = min(n, Int(round(i + threeish_sigma(σs[i, order], bhws[i, order]))))
            holder[i, lo:hi] = conv_gauss_tophat.((lo-i):(hi-i), σs[i, order], bhws[i, order])
            # holder[i, lo:hi] = conv_gauss_tophat_integral.(σs[i, order], bhws[i, order], (lo-i):(hi-i))
            holder[i, lo:hi] ./= sum(view(holder, i, lo:hi))
        end
        ans = sparse(holder)
        dropzeros!(ans)
        return ans
    end

    # KM: Adding the option to extrapolate
    function neid_lsf(order::Int, log_λ_neid_order::AbstractVector, log_λ_obs::AbstractVector; extrapolate::Bool=false)
        @assert 1 <= order <= length(no_lsf_orders)
        if no_lsf_orders[order]; return nothing end
        n = length(log_λ_obs)

        # need to convert σs, bhws, and threeish_sigma (in units of neid pixels) to units of log_λ_obs pixels
        # pixel_separation_log_λ_obs = linear_interpolation(log_λ_obs, SSOF.simple_derivative(log_λ_obs); extrapolation_bc=Line())
        pixel_separation_log_λ_obs = DataInterpolations.LinearInterpolation(SSOF.simple_derivative(log_λ_obs), log_λ_obs)
        pixel_separation_ratio = SSOF.simple_derivative(log_λ_neid_order) ./ pixel_separation_log_λ_obs.(log_λ_neid_order)
        # make the linear_interpolation object and evaluate it
        # converter(vals) = linear_interpolation(log_λ_neid_order, pixel_separation_ratio .* vals; extrapolation_bc=Line())(log_λ_obs)
        converter(vals) = (DataInterpolations.LinearInterpolation(pixel_separation_ratio .* vals, log_λ_neid_order; extrapolate=extrapolate)).(log_λ_obs)
        σs_converted = converter(σs[:, order])
        bhws_converted = converter(bhws[:, order])
        threeish_sigma_converted = converter(threeish_sigma.(σs[:, order], bhws[:, order]))

        holder = zeros(n, n)
        for i in 1:n
            lo = max(1, Int(round(i - threeish_sigma_converted[i])))
            hi = min(n, Int(round(i + threeish_sigma_converted[i])))
            holder[i, lo:hi] = conv_gauss_tophat.((lo-i):(hi-i), σs_converted[i], bhws_converted[i])
            holder[i, lo:hi] ./= sum(view(holder, i, lo:hi))
        end
        ans = sparse(holder)
        dropzeros!(ans)
        return ans
    end


end # module

# s = NEID_lsf(100)
# heatmap(Matrix(s[1:100,1:100]))
# heatmap(Matrix(s[end-100:end,end-100:end]))
#
# avg_nz_pix_neighbors = Int(round(length(s.nzval)/s.n/2))
# i = 100
# xx = (i-avg_nz_pix_neighbors-5):(i+avg_nz_pix_neighbors+5)
# plot_subsection = s[i, xx]
# plot(xx, plot_subsection)
# plot!(xx, iszero.(plot_subsection)./10)
# vline!([i])
