module EXPRESLSF
    import StellarSpectraObservationFitting as SSOF
    # SSOF_path = dirname(dirname(pathof(SSOF)))
    # include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
    # SSOFU = SSOFUtilities

    ## Finding LSF width as a function of wavenumber
    using Distributions
    using SpecialFunctions
    using SparseArrays
    using LinearAlgebra
    using JLD2

    filename = "EXPRES/expres_lsf.jld2"

    if isfile(filename)
        @load filename _w_intra _w_inter _min_order _max_order
    else
        poly_order = 2
        using CSV, DataFrames
        eo = CSV.read("EXPRES/expres_psf.txt", DataFrame)
        # eo = CSV.read("C:/Users/chris/Downloads/expres_psf.txt", DataFrame)
        filter!(:line => ==("LFC"), eo)
        sort!(eo, ["wavenumber [1/cm]"])

        unit_str = "1/cm"
        xlab = "Wavenumber ($unit_str)"
        wns = eo."wavenumber [1/cm]"
        σ = SSOF.fwhm_2_σ.(eo."fwhm [1/cm]")

        orders = minimum(eo.order)+1:maximum(eo.order)-1
        @assert poly_order > 0

        _w_intra = Array{Float64}(undef, 3, length(orders))
        min_wns = Array{Float64}(undef, length(orders))
        min_σs = Array{Float64}(undef, length(orders))
        for i in 1:length(orders)
            ord = orders[i]
            df = filter(:order => ==(ord), eo)
            _dm = ones(nrow(df), 1 + poly_order)
            _dm[:, 2] = df."wavenumber [1/cm]"
            for i in 2:poly_order
                _dm[:, i + 1] = _dm[:, 2] .^ i
            end
            _σ = SSOF.fwhm_2_σ.(df."fwhm [1/cm]")
            _w_intra[:, i] = SSOF.general_lst_sq(_dm, _σ, (df."fwhm error [1/cm]") .^ 2)  # note the errors are not scaled FWHM -> σ
            min_ind = argmin(df."fwhm [1/cm]")
            min_wns[i] = _dm[min_ind, 2]
            min_σs[i] = _σ[min_ind]
        end
        dm = ones(length(orders), 1 + poly_order)
        dm[:, 2] = min_wns
        for i in 2:poly_order
            dm[:, i + 1] = dm[:, 2] .^ i
        end
        _w_inter = SSOF.general_lst_sq(dm, min_σs)
        _min_order, _max_order = extrema(orders)
        @save filename _w_intra _w_inter _min_order _max_order
    end

    function lsf_σ(wn::AbstractVector, order::Int)
        dm = ones(length(wn), length(_w_inter))
        dm[:, 2] = wn
        for i in 2:length(_w_inter)-1
            dm[:, i+1] = wn .^ i
        end
        if _min_order <= order <= _max_order
            return dm * _w_intra[:, order+1-_min_order]
        else
            return dm * _w_inter
            # return nothing
        end
    end

    function gaussian_integral(σ::Real, x1mμ::Real, x2mμ::Real)
        factor = 1 / (sqrt(2) * σ)
        return (erf(x2mμ * factor) - erf(x1mμ * factor)) / 2
    end

    function expres_lsf(λ::AbstractVector, order::Int; safe::Bool=true)
        if safe; @assert 1000 < mean(λ) < 50000  "Are you sure you're using λ (Å) and not wavenumber (1/cm) or log(λ)?" end
        wn = SSOF.Å_to_wavenumber.(λ)
        σs = lsf_σ(wn, order)
        if isnothing(σs); return nothing end
        nwn = -wn
        holder = zeros(length(nwn), length(nwn))
        # max_w = 0
        for i in eachindex(nwn)
            lo, hi = SSOF.searchsortednearest(nwn, [nwn[i] - 3.5 * σs[i], nwn[i] + 3.5 * σs[i]])
            holder[i, lo:hi] = pdf.(Normal(wn[i], σs[i]), wn[lo:hi])
            # if lo == 1
            #     geo_mean = Array{Float64}(undef, hi-lo+2)
            #     geo_mean[2:end] = sqrt.(view(wn, lo:hi).*view(wn,lo+1:hi+1)) .- wn[i]
            #     geo_mean[1] = 2*geo_mean[1]-geo_mean[2]
            # elseif hi == length(wn)
            #     geo_mean = Array{Float64}(undef, hi-lo+2)
            #     geo_mean[1:end-1] = sqrt.(view(wn, lo-1:hi-1).*view(wn,lo:hi)) .- wn[i]
            #     geo_mean[end] = 2*geo_mean[end-1]-geo_mean[end-2]
            # else
            #     geo_mean = sqrt.(view(wn, lo-1:hi).*view(wn,lo:hi+1)) .- wn[i]
            # end
            # holder[i, lo:hi] = gaussian_integral.(σs[i], view(geo_mean, 1:length(geo_mean)-1), view(geo_mean, 2:length(geo_mean)))
            holder[i, lo:hi] ./= sum(view(holder, i, lo:hi))
            # max_w = max(max_w, max(hi-i, i-lo))
        end
        ans = sparse(holder)
        dropzeros!(ans)
        return ans
        # return BandedMatrix(holder, (max_w, max_w))
    end
    # EXPRES_lsfs(λ::AbstractMatrix; kwargs...) =
    #     [lsf(view(λ, :, i); kwargs...) for i in 1:size(λ, 2)]
    expres_lsf(λ::AbstractMatrix, order::Int; kwargs...) =
        expres_lsf(vec(median(λ; dims=2)), order; kwargs...)
end

# xx = EXPRESLSF.EXPRESLSF
# plt = SSOFU._plot(; xlabel=xx.xlab, ylabel="LSF σ ($(xx.unit_str))")
# scatter!(plt, xx.wns[:], xx.σ[:]; label="", xlabel=xx.xlab, ylabel="LSF σ ($(xx.unit_str))", markerstrokewidth = 0.5)
# for order in xx.orders
#     inds_temp = xx.eo.order .== order
#     plot!(plt, xx.wns[inds_temp], xx.lsf_σ(xx.wns[inds_temp], order); label="", lw=4)
# end
# plot!(xx.wns[:], xx.lsf_σ(xx.wns, 0); label="", lw=4)
# png(plt, "expres_lsf_model")
#
# plt = SSOFU._plot(; xlabel=xx.xlab, ylabel="LSF σ ($(xx.unit_str))")
# for order in [38,39]
#     inds_temp = xx.eo.order .== order
#     scatter!(plt, xx.wns[inds_temp], xx.σ[inds_temp]; label="")
#     plot!(plt, xx.wns[inds_temp], xx.lsf_σ(xx.wns[inds_temp], order); label="", lw=4)
# end
# plt
# png(plt, "expres_lsf_model_zoom")
