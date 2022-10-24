

############
# code shamelessly ripped and modified from https://github.com/RvSpectML/NeidSolarScripts.jl/blob/fc6db7979b649370a2923ac987dbd02424cb1762/src/continuum_rassine_like.jl
##########
# ] add DSP DataInterpolations Distributions NaNMath Polynomials RollingFunctions SortFilters

# using RvSpectMLBase
using Statistics, NaNMath # For mean, median, etc
using DSP                                 # For smoothing flux
using DataInterpolations    # For predicting continuum between anchors
using RollingFunctions        # For rolling maximum.    Used in setting radius of rolling-pin.
using SortFilters                 # For rolling quantile for fitting polynomial while avoiding lines + computing IQR for sigma clipping.    https://github.com/sairus7/SortFilters.jl
# using FITSIO                            # For reading SED
using Polynomials                 # For fitting out low-order polynomial

# constants
speed_of_light_mks = 299792458    # m/s    TODO: Update number
fwhm_sol = 7.9e3                    # m/s    TODO: Update number

function calc_mean_snr( flux::AV1, var::AV2 ) where { T1<:Real, T2<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2} }
     idx_bad = isnan.(flux) .| isnan.(var) .| (var .<=0.0)
     return mean(flux[.!idx_bad]./sqrt.(var[.!idx_bad]))
end

function smooth(f::AV2; half_width::Integer = 6 ) where {    T2<:Real, AV2<:AbstractVector{T2} }
    #println("# size(f) to smooth: ", size(f))
    w=DSP.hanning(1+2*half_width)
    if 1+2*half_width<3
        return f
    elseif length(f)<1+2*half_width
        return f
    else
        return y_smooth = conv(f, w/sum(w))[1+half_width:end-half_width]
    end
end

function calc_rolling_max(f::AV2; width::Integer = 13 ) where {    T2<:Real, AV2<:AbstractVector{T2} }
    #shift = max(1,floor(Int64,width//2))
    @assert width%2 == 1
    shift = max(1,convert(Int64,(width+1)//2))
    #println(" size(f) = ",size(f), " width= ", width, " shift = ", shift, " rolling_max = ", size(rolling(NaNMath.maximum,f,width)), " target = ", size((shift+1):(length(f)-shift)))
    z = similar(f)
    #z[(shift):(length(f)-shift+1)] .= rollmax(f,width)
    z[(shift):(end-shift+1)] .= rolling(NaNMath.maximum,f,width)
    z[(end-shift+2):end] .= z[end-shift+1]
    z[1:shift-1] .= z[shift]
    return z
end

function find_local_maxima(f::AV2; half_width::Integer = 6 ) where {    T2<:Real, AV2<:AbstractVector{T2} }
    rollingmax = calc_rolling_max(f,width=half_width*2+1)
    findall(f .== rollingmax)
end

function longest_cluster_same_sign(y::AV) where { T<:Real, AV<:AbstractVector{T} }
    longest = 0
    current = 1
    this_start = 1
    start = 1
    stop = 1
    sign_last = sign(first(y))
    for i in 2:length(y)
        sign_this = sign(y[i])
        if sign_this == sign_last
            current += 1
        else
            if current > longest
                longest = current
                start = this_start
                stop = i-1
            end
            current = 1
            this_start = i
        end
        sign_last = sign_this
    end
    if current > longest
        longest = current
        start = this_start
        stop = length(y)
    end
    return (len=longest, start=start, stop=stop)
end

function calc_rollingpin_radius(λ::AV1, f::AV2; fwhm::Real = fwhm_sol,
                        S1_width_factor::Real = 40, S2_width_factor::Real = 10,
                        min_R_factor::Real = 100, ν::Real = 1, verbose::Bool = false ) where { T1<:Real, AV1<:AbstractVector{T1}, T2<:Real, AV2<:AbstractVector{T2}    }
    λlo = minimum(λ)
    S1_width = S1_width_factor*fwhm/speed_of_light_mks*λlo
    S1_width = min(2*floor(Int64,S1_width/2),2)+1
    S2_width = S2_width_factor*S1_width
    S2_width = 2*floor(Int64,S2_width/2)+1
    f_max1 = calc_rolling_max(f,width=S1_width)
    f_max2 = calc_rolling_max(f,width=S2_width)
    (longest_len, longest_start, longest_stop) = longest_cluster_same_sign(f_max2.-f_max1)
    longest_Δλoverλ = (λ[longest_stop]-λ[longest_start]) / ((λ[longest_start]+λ[longest_stop])/2)
    #penalty = (f_max2.-f_max1)./f_max2
    #conversion_fwhm_sig = (10*min_λ/(sqrt(8*log(2))*speed_of_light_kmps))
    #min_radius = fwhm * conversion_fwhm_sig # par_R
    #radius = min_radius .* λ./min_λ
    rollingpin_R_min = min_R_factor*λlo*(fwhm/speed_of_light_mks)/log(8*log(2))
    #rollingpin_R_max = max_R_factor*rollingpin_R_min    # Arbitrary, not same as Rassine's auto setting, see https://github.com/MichaelCretignier/Rassine_public/blob/master/Rassine.py#L718
    rollingpin_R_max = max(λlo*longest_Δλoverλ,rollingpin_R_min*2)
    #rollingpin_R_min = 2*floor(Int64,rollingpin_R_min/2)
    #rollingpin_R_max = 2*floor(Int64,rollingpin_R_max/2)
    if verbose
        println("longest_len = ", longest_len)
        println("longest_Δλoverλ = ", longest_Δλoverλ)
        println("rollingpin_R_min = ", rollingpin_R_min)
        println("rollingpin_R_max = ", rollingpin_R_max)
    end
    rollingpin_radius = (λ./λlo).*(rollingpin_R_min .+ (rollingpin_R_max-rollingpin_R_min).*((f_max2.-f_max1)./f_max2).^ν)
    #plot!(lambda,f_max1,color=:red)
    #plot!(lambda,f_max2,color=:magenta)
    #plot!(lambda,penalty)
    return rollingpin_radius
end

function calc_continuum_anchors(λ::AV1, f::AV2; radius::AV3,
                        stretch_factor::Real = 1.0, fwhm::Real = 7.3 #= km/s=#,
                                                verbose::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3} }
    @assert length(λ) == length(f)
    #speed_of_light_kmps = 3e5
    #par_stretching = 2# 0.5# 0.5 #11.385
    extrema_λ = extrema(λ)
    min_λ = first(extrema_λ)
    extrema_f = extrema(f)
    #normalization = (extrema_f[2]-extrema_f[1])/(extrema_λ[2]-extrema_λ[1])
    quantile_for_extrema = max(0.5,(length(f) >= 4000) ? 0.995 : 1-20/length(f))
    (min_f, max_f) = quantile(Iterators.filter(!isnan, f),[1-quantile_for_extrema,quantile_for_extrema])
    min_f = 0.0
    normalization = (max_f-min_f)/(extrema_λ[2]-extrema_λ[1])
    normalization *= stretch_factor
    function calc_dist(i::Integer, j::Integer)    # TODO OPT: eliminate sqrt
        if λ[i]<=λ[j]
            return 0.0
        else
            sqrt((λ[i]-λ[j])^2 +((f[i]-f[j])/normalization)^2)
        end
    end
    numero = range(1,stop=length(λ))
    mask = falses(length(λ))
    dist = zeros(length(λ))
    #conversion_fwhm_sig = (10*min_λ/(sqrt(8*log(2))*speed_of_light_kmps))
    #min_radius = fwhm * conversion_fwhm_sig # par_R
    #println("# old min_radius = ", min_radius)
    if verbose println("# passed radius = ", extrema(radius)) end
    #radius = min_radius .* λ./min_λ
    keep = Vector{Int64}()
    sizehint!(keep,max(32,min(256,2^floor(Int64,log2(length(λ))-2))))
    push!(keep,1)
    j = 1
    while length(λ)-j>2
        par_R = radius[j]
        if j==1 par_R /= 1.5 end
        map!(i->calc_dist(i,j),dist,1:length(λ))
        #mask .= (0.0 .< distance[:,j].<2.0*par_R)
        mask .= (0.0 .< dist.<2.0*par_R)
        while sum(mask)==0
            par_R *= 1.5
            #mask .= (0.0 .< distance[:,j].<2.0*par_R)
            mask .= (0.0 .< dist.<2.0*par_R)
        end
        p1 = [ λ[j] f[j]/normalization ]
        p2 = hcat(λ[mask], f[mask]./normalization)
        delta = p2 .- p1
        #c = sqrt.(delta[:,1].^2 .+delta[:,2].^2)
        c = sqrt.(vec(sum(delta.^2,dims=2)))
        harg = (par_R.^2 .-0.25.*c.^2)
        #=
        if any(harg.<0)
            println("# j = ", j, " dist = ", distance[j:j+10,j])
            println("# R^2 = ", par_R^2)
            println("# c^2/4 = ", c.^2/4)
            println("# harg = ", harg)
        end
        h = zeros(length(harg))
        hmask = harg.>0
        h[hmask] .= sqrt.(harg[hmask])
        =#
        h = sqrt.(harg)
        cx = p1[1] .+ 0.5.*delta[:,1] .-h./c.*delta[:,2]
        cy = p1[2] .+ 0.5.*delta[:,2] .+h./c.*delta[:,1]
        #return (cx,cy)
        #cond1 = (cy.-p1[2]).>=0
        #theta = zeros(length(cy))
        mintheta = Inf
        argmintheta = 0
        for i in 1:length(cy)
            if cy[i]-p1[2] >0
                acos_arg = (cx[i]-p1[1])/par_R
                if acos_arg > 1.0 acos_arg = 1.0
                elseif    acos_arg < -1.0 acos_arg = -1.0    end
                thetai = -acos(acos_arg)+π
            else
                asin_arg = (cy[i]-p1[2])/par_R
                if asin_arg > 1.0 asin_arg = 1.0
                elseif    asin_arg < -1.0 asin_arg = -1.0    end
                thetai = -asin(asin_arg)+π
            end
            if thetai < mintheta
                mintheta = thetai
                argmintheta = i
            end
        end
        #theta[cond1] .= -acos.((cx[cond1].-p1[1])./par_R).+π
        #theta[.!cond1] .= -asin.((cy[.!cond1].-p1[2])./par_R).+π
        #j2 = argmin(theta)
        j2 = argmintheta
        j = numero[mask][j2]
        push!(keep,j)
    end
    if verbose println("# using ", length(keep), " anchor points") end
    return keep
end

function calc_continuum_from_anchors( λ::AV1, f::AV2, anchors::AV3; λout::AV4 = λ,
                            verbose::Bool = false ) where {
                                T1<:Real, T2<:Real, T3<:Integer, T4<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}    }
    calc_continuum_from_anchors_hybrid( λ, f, anchors, λout=λout,verbose=verbose)
end

function calc_continuum_from_anchors_linear( λ::AV1, f::AV2, anchors::AV3; λout::AV4 = λ,
                            verbose::Bool = false ) where {
                                T1<:Real, T2<:Real, T3<:Integer, T4<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}    }
    @assert length(anchors) >= 2
    #if length(anchors) <= 1     return ones(size(λout))    end
    interp_linear = LinearInterpolation(f[anchors],λ[anchors])
    function extrap(x::Real)
        if x<first(interp_linear.t)
            slope = (interp_linear.u[2]-interp_linear.u[1])/(interp_linear.t[2]-interp_linear.t[1])
            return first(interp_linear.u)+(x-interp_linear.t[1])*slope
        elseif x>last(interp_linear.t)
            slope = (interp_linear.u[end]-interp_linear.u[end-1])/(interp_linear.t[end]-interp_linear.t[end-1])
            return last(interp_linear.u)+(x-interp_linear.t[end])*slope
        else
            return interp_linear(x)
        end
    end
    return extrap.(λout)
end

function calc_continuum_from_anchors_cubic( λ::AV1, f::AV2, anchors::AV3; λout::AV4 = λ,
                            verbose::Bool = false ) where {
                                T1<:Real, T2<:Real, T3<:Integer, T4<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}    }
    interp_cubic = CubicSpline(f[anchors],λ[anchors])
    interp_cubic.(λout)
end


function calc_continuum_from_anchors_hybrid( λ::AV1, f::AV2, anchors::AV3; λout::AV4 = λ,
                            verbose::Bool = false ) where {
                                T1<:Real, T2<:Real, T3<:Integer, T4<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}    }
    output = similar(λout)
    if length(anchors) < 2
        output .= ones(size(λout))
    elseif length(anchors) < 9
        output .= calc_continuum_from_anchors_linear( λ, f, anchors, λout=λout,verbose=verbose)
    else
        λmin = λ[anchors[4]]
        λmax = λ[anchors[end-3]]
        idx_pix_cubic = searchsortedfirst(λout,λmin):searchsortedlast(λout,λmax)
        output[idx_pix_cubic] .= calc_continuum_from_anchors_cubic( λ, f, anchors, λout=λout[idx_pix_cubic],verbose=verbose)
        idx_pix_linear1 = 1:first(idx_pix_cubic)
        idx_pix_linear2 = last(idx_pix_cubic):length(λout)
        idx_pix_linear = vcat(idx_pix_linear1,idx_pix_linear2)
        if verbose println("# idx_pix_linear = ", idx_pix_linear, " \n # λ_linear = ", λout[idx_pix_linear]) end
        output[idx_pix_linear] .= calc_continuum_from_anchors_linear( λ, f, anchors, λout=λout[idx_pix_linear],verbose=verbose)
    end
    return output
end

function replace_edge_anchor_vals!(f::AV1, n::Integer = 3 ) where { T1<:Real, AV1<:AbstractVector{T1} }
    @assert length(f) >= 2*n+1
    f[1:(n+1)] .= f[n+1]
    f[end-n:end] .= f[length(f)-n-1]
    return f
end

function find_clean_anchors_by_slope(anchors::AV1, f::AV2; threshold::Real = 0.995, verbose::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3} }
    nanchors = length(anchors)
    @assert nanchors == length(f)
    @assert nanchors >= 8
    Δabsslope = zeros(length(anchors))
    for i in 2:nanchors-1
        slope_hi = (f[i+1]-f[i])/(anchors[i+1]-anchors[i])
        slope_lo = (f[i]-f[i-1])/(anchors[i]-anchors[i-1])
        Δabsslope[i] = abs(slope_hi-slope_lo)
    end
    threshold = quantile(Δabsslope[2:end-1],threshold)
    mask = Δabsslope.<threshold
    return mask
end

function merge_nearby_anchors(anchors::AV1, λ::AV2; threshold::Real = 1, verbose::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3} }
    Δλ = λ[anchors[2:end]] .- λ[anchors[1:end-1]]
    close_anchor_pair_mask = Δλ .<= threshold
    nclose = sum(close_anchor_pair_mask)
    if verbose println("# Found ", nclose, " close anchor pairs out of ", length(anchors), ".")    end
    if nclose == 0
        return anchors
    else
        if verbose
            println("# Anchors = ", anchors)
            println("Close anchor pairs: ", findall(close_anchor_pair_mask), " => ", anchors[findall(close_anchor_pair_mask)])
        end
    end
    old_anchor_mask = falses(length(anchors))
    if first(Δλ) > threshold
        old_anchor_mask[1] = true
    end
    for i in 2:length(Δλ)
        if (Δλ[i-1] > threshold) && (Δλ[i] > threshold)
            old_anchor_mask[i] = true
        end
    end
    if last(Δλ) > threshold
        old_anchor_mask[length(anchors)] = true
    end
    old_anchors = anchors[old_anchor_mask]
    if verbose     println("# Old anchors = ", old_anchors)    end
    close_anchor_idx = (1:(length(anchors)-1))[close_anchor_pair_mask]
    new_anchors = zeros(Int64,nclose)
    resize!(new_anchors,0)
    idx_start = 1
    for i in 2:nclose
        if close_anchor_idx[i]-close_anchor_idx[i-1] == 1
            continue
        else
            idx_stop = i-1

            if idx_start == idx_stop
                merged_anchor_idx = idx_start
            elseif (idx_stop-idx_start)%2 == 0
                merged_anchor_idx = idx_start + convert(Int64,(idx_stop-idx_start)//2)
            else
                merged_anchor_idx = idx_start + convert(Int64,(idx_stop-idx_start+1)//2)
            end
            merged_anchor = anchors[close_anchor_idx[merged_anchor_idx]]
            push!(new_anchors,merged_anchor)
            idx_start = i
        end
    end
    if idx_start == nclose == 1
        push!(new_anchors,anchors[close_anchor_idx[idx_start]])
    elseif idx_start != nclose
        idx_stop = nclose
        if idx_start == idx_stop
            merged_anchor_idx = idx_start
        elseif (idx_stop-idx_start)%2 == 0
            merged_anchor_idx = idx_start + convert(Int64,(idx_stop-idx_start)//2)
        else
            merged_anchor_idx = idx_start + convert(Int64,(idx_stop-idx_start+1)//2)
        end
        merged_anchor = anchors[close_anchor_idx[merged_anchor_idx]]
        push!(new_anchors,merged_anchor)
    end
    if verbose println("# Keeping ", length(new_anchors), " new anchors, ", length(old_anchors), " old anchors.")    end
    anchors_out = mergesorted(old_anchors,new_anchors)
    return anchors_out
end

function calc_continuum(λ::AV1, f_obs::AV2, var_obs::AV3; λout::AV4 = λ, fwhm::Real = fwhm_sol, ν::Real =1.0,
    smoothing_half_width::Integer = 6, local_maximum_half_width::Integer = smoothing_half_width+1,
     stretch_factor::Real = 5.0, merging_threshold::Real = 0.25,    min_R_factor::Real = 100.0, smoothing_half_width_no_anchors::Integer = 1000, verbose::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real, T4<:Real, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}    }
    #clip_width_A = 1.0
    #clip_width_pix = clip_width_A/(λ[2]-λ[1])
    #@assert 1 <= clip_width_pix < Inf
    #clip_width_pix = 2*floor(Int64,clip_width_pix/2)
    mean_snr_per_pix = calc_mean_snr(f_obs,var_obs)
    #smoothing_half_width = (mean_snr_per_pix >= 30) ? smoothing_half_width : 40
    if mean_snr_per_pix < 20
        smoothing_half_width = min(100, ceil(Int64,6*(20/mean_snr_per_pix)^2))
    end
    f_smooth = smooth(f_obs, half_width=smoothing_half_width)
    idx_local_maxima = find_local_maxima(f_smooth, half_width=local_maximum_half_width)
    if verbose println("# Found ", length(idx_local_maxima), " local maxima." )    end
    if length(idx_local_maxima) < 7
        println("# Warning only ",    length(idx_local_maxima), " local maxima, aborting order.")
        half_width = min_R_factor*(fwhm/speed_of_light_mks)/log(8*log(2))*minimum(λ)/(λ[2]-λ[1])
        f_alt_continuum = smooth(f_obs, half_width=floor(Int64,half_width/2)*2 )
        return (anchors=Int64[], continuum=f_alt_continuum, f_filtered=f_smooth)
    end
    #(f_median_filtered, f_clip_threshold, f_clean) = calc_rolling_median_and_max_to_clip(λ,f_obs, width=clip_width_pix, verbose=verbose)
    rollingpin_radius = calc_rollingpin_radius(λ, f_smooth, fwhm=fwhm, min_R_factor=min_R_factor, verbose=verbose, ν=ν)

    anch_orig = calc_continuum_anchors(λ[idx_local_maxima],f_smooth[idx_local_maxima],radius=rollingpin_radius[idx_local_maxima], stretch_factor=stretch_factor, verbose=verbose)
    if verbose println("# Found ", length(anch_orig), " potential anchors." )    end
    anch_orig = idx_local_maxima[anch_orig]
    anchor_vals = f_smooth[anch_orig]
    if length(anch_orig) >= 8    # 2n+1, for replace_edge_anchor_vals!
        replace_edge_anchor_vals!(anchor_vals)
        anch_mask = find_clean_anchors_by_slope(anch_orig,anchor_vals, threshold = 0.95, verbose=verbose)
        if verbose println("# After rejected high-slope anchors ", sum(anch_mask), " anchors left." )    end
    else
        anch_mask = trues(length(anch_orig))
    end
    #anchors_merged = anch_orig[anch_mask]
    if merging_threshold > 0    && length(anch_orig) >= 4
        anchors_merged = merge_nearby_anchors(anch_orig[anch_mask],λ, threshold=merging_threshold, verbose=verbose)
        if verbose println("# After merging ", length(anchors_merged), " anchors left." )    end
    else
        anchors_merged = anch_orig[anch_mask]
    end
    if length(anchors_merged) < 4
        println("# Warning only ",    length(anchors_merged), " anchors found, using simpler smoothing.")
        #half_width = min_R_factor*(fwhm/speed_of_light_mks)/log(8*log(2))*minimum(λ)/(λ[2]-λ[1])
        #f_alt_continuum = smooth(f_obs, half_width=floor(Int64,half_width/2)*2 )
        f_alt_continuum = smooth(f_obs, half_width=smoothing_half_width_no_anchors)
        return (anchors=Int64[], continuum=f_alt_continuum, f_filtered=f_smooth)
    end
    #anchors_merged = anch_orig[anch_mask]
    continuum = calc_continuum_from_anchors_hybrid(λ,f_smooth,anchors_merged, λout=λout) # , verbose=verbose)
    return (anchors=anchors_merged, continuum=continuum, f_filtered=f_smooth)
end

function calc_continuum(λ::AV1, f_obs::AV2, var_obs::AV3, anchors::AV5; λout::AV4 = λ, smoothing_half_width::Integer = 6, smoothing_half_width_no_anchors::Integer = 1000, verbose::Bool = false ) where { T1<:Real, T2<:Real, T3<:Real, T4<:Real, T5<:Integer, AV1<:AbstractVector{T1}, AV2<:AbstractVector{T2}, AV3<:AbstractVector{T3}, AV4<:AbstractVector{T4}, AV5<:AbstractVector{T5}    }
    #clip_width_A = 1.0
    #clip_width_pix = clip_width_A/(λ[2]-λ[1])
    #@assert 1 <= clip_width_pix < Inf
    #clip_width_pix = 2*floor(Int64,clip_width_pix/2)
    mean_snr_per_pix = calc_mean_snr(f_obs,var_obs)
    #smoothing_half_width = (mean_snr_per_pix >= 30) ? smoothing_half_width : 40
    if mean_snr_per_pix < 30
        smoothing_half_width = min(100, ceil(Int64,6*(30/mean_snr_per_pix)^2))
    end
    f_smooth = smooth(f_obs, half_width=smoothing_half_width)
    if length(anchors) < 4
        continuum = smooth(f_obs, half_width=smoothing_half_width_no_anchors)
    else
        continuum = calc_continuum_from_anchors_hybrid(λ,f_smooth,anchors, λout=λout) # , verbose=verbose)
    end
    return (anchors=anchors, continuum=continuum, f_filtered=f_smooth)
end

function calc_continuum(λ::AA1, f_obs::AA2, var_obs::AA3; λout::AA4 = λ, fwhm::Real = fwhm_sol, ν::Real =1.0,
    stretch_factor::Real = 5.0, merging_threshold::Real = 0.25, smoothing_half_width::Integer = 6, min_R_factor::Real = 100.0, smoothing_half_width_no_anchors::Integer = 1000,
                orders_to_use::AbstractVector{<:Integer} = 1:size(λ,2), verbose::Bool = false ) where {
                T1<:Real, T2<:Real, T3<:Real, T4<:Real, AA1<:AbstractArray{T1,2}, AA2<:AbstractArray{T2,2}, AA3<:AbstractArray{T3,2} , AA4<:AbstractArray{T4,2} }
    @assert size(λ) == size(f_obs) == size(var_obs)
    @assert size(λout,2) == size(λout,2)
    num_orders = size(λout,2)
    anchors_2d = fill(Int64[],num_orders)
    continuum_2d = fill(NaN,size(λout,1),size(λout,2))
    f_filtered_2d = fill(NaN,size(λout,1),size(λout,2))
    #Threads.@threads for ord_idx in orders_to_use
    for ord_idx in orders_to_use
        if verbose
            println("# Order index = ", ord_idx)
            flush(stdout)
        end
        pix = isfinite.(view(var_obs,:,ord_idx))
        λ_use = view(λ,pix,ord_idx)
        f_obs_use = convert.(Float64,view(f_obs,pix,ord_idx))
        var_obs_use = convert.(Float64,view(var_obs,pix,ord_idx))
        if all(isnan.(λ_use)) || all(isnan.(f_obs_use)) || all(isnan.(var_obs_use))     continue     end
        λout_use = view(λout,pix,ord_idx)
        (anchors_1d, continuum_order_1d, f_filtered_1d) = calc_continuum(λ_use,f_obs_use,var_obs_use, λout=λout_use,
                 stretch_factor=stretch_factor, merging_threshold=merging_threshold, smoothing_half_width=smoothing_half_width, min_R_factor=min_R_factor, smoothing_half_width_no_anchors=smoothing_half_width_no_anchors, verbose=verbose)
        anchors_2d[ord_idx] = anchors_1d
        continuum_2d[pix,ord_idx] .= continuum_order_1d
        f_filtered_2d[pix,ord_idx] .= f_filtered_1d
    end
    return (anchors=anchors_2d, continuum=continuum_2d, f_filtered=f_filtered_2d)
end

function mergesorted(l1::AbstractVector, l2::AbstractVector)
    ll1 = length(l1)
    ll2 = length(l2)
    if length(l1)==0
        return l2
    elseif length(l2)==0
        return l1
    end
    result = similar(promote_type(typeof(l1),typeof(l2)), ll1+ll2)
    i = j = k = 1
    while i <= ll1 && j <= ll2
        if l1[i] <= l2[j]
            result[k] = l1[i]
            i += 1
        else
            result[k] = l2[j]
            j += 1
        end
        k += 1
    end
    if i <= ll1
        result[k:end] .= l1[i:end]
    elseif j <= ll2
        result[k:end] .= l2[j:end]
    end
    return result
end
