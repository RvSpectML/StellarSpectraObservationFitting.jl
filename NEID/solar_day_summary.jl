## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using Statistics
using JLD2
using Optim
using Nabla
using StatsBase

## Setting up necessary variables

# using DataFrames, CSV
# using Dates
# hmm3 = CSV.read("D:/Christian/Downloads/summary_1.csv", DataFrame)
# hmm4 = filter(row -> row.num_rvs_good > 150, hmm3)
# dates = Date.(hmm4.obs_date_string, dateformat"m/d/y")
# stars = Dates.format.(dates, dateformat"yyyy/mm/dd")

stars = ["2021/01/30", "2021/02/04", "2021/02/06", "2021/02/08", "2021/02/11", "2021/02/12", "2021/02/14", "2021/02/18", "2021/02/20", "2021/02/21", "2021/02/22", "2021/02/24", "2021/02/25", "2021/03/01", "2021/03/02", "2021/03/05", "2021/03/06", "2021/03/10", "2021/03/14", "2021/03/17", "2021/03/20", "2021/03/22", "2021/03/27", "2021/03/28", "2021/03/30", "2021/03/31", "2021/04/02", "2021/04/03", "2021/04/04", "2021/04/13", "2021/04/17", "2021/04/18", "2021/04/19", "2021/04/21", "2021/04/23", "2021/04/24", "2021/04/29", "2021/04/30", "2021/05/03", "2021/05/04", "2021/05/06", "2021/05/07", "2021/05/10", "2021/05/11", "2021/05/12", "2021/05/14", "2021/05/16", "2021/05/17", "2021/05/19", "2021/05/21", "2021/05/22", "2021/05/27", "2021/05/28", "2021/05/29", "2021/05/30", "2021/05/31", "2021/06/01", "2021/06/02", "2021/06/03", "2021/06/05", "2021/06/06", "2021/06/11", "2021/06/12", "2021/06/13", "2021/06/14", "2021/06/19", "2021/06/20", "2021/06/21", "2021/06/25", "2021/06/27", "2021/07/10", "2021/07/11", "2021/07/13", "2021/07/28", "2021/08/02", "2021/09/20", "2021/09/21", "2021/09/28", "2021/10/28", "2021/10/29", "2021/10/30", "2021/10/31", "2021/11/03", "2021/11/04", "2021/11/06", "2021/11/08", "2021/11/10", "2021/11/11", "2021/11/12", "2021/11/13", "2021/11/14", "2021/11/15", "2021/11/18", "2021/11/26", "2021/11/27", "2021/11/28", "2021/11/29", "2021/12/03"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines expres_data_path and expres_save_path

loss(model, interps, data) =
    sum(((SSOF.spectra_interp(model, interps) .- data.flux) .^ 2) ./ data.var)

function optim_obj(interps, data, init)
    f(model) = loss(model, interps, data)
    g = ∇(f)
    g_val = ∇(f; get_output=true)
    g(init)  # compile it
    g_val(init)  # compile it
    function g!(G, θ)
        G[:] = only(g(θ))
    end
    function fg_obj!(G, θ)
        l, g = g_val(θ)
        G[:] = only(g)
        return l.val
    end
    obj = OnceDifferentiable(f, g!, fg_obj!, init)
    return obj
end

function fit_var(optim, data_base, interps; n::Int=30)
    data_holder = copy(data_base)
    results_holder = ones(n, length(optim))
    data_holder.var[data_holder.var.==Inf] .= 0
    data_noise = sqrt.(data_holder.var)
    data_holder.var[data_holder.var.==0] .= Inf

    for i in 1:n
        data_holder.flux .= data_base.flux .+ (data_noise .* randn(size(data_base.var)))
        result = Optim.optimize(optim_obj(interps, data_holder, optim), optim, LBFGS())
        results_holder[i, :] =  result.minimizer
    end
    return vec(std(results_holder; dims=1)) .^ 2
end

# for desired_order in 7:118
for desired_order in 60:85
    base_path = neid_save_path * star * "/$(desired_order)/"
    data_path = base_path * "data.jld2"
    save_path = base_path * "summary.jld2"

    if isfile(data_path)

        # load data
        @load data_path n_obs data times_nu airmasses

        # daily summary λs
        log_λ, _ = SSOF.create_λ_template(data.log_λ_obs)
        log_λ_star = log_λ .- mean(data.log_λ_obs .- data.log_λ_star)

        # get initial daily_flux guess
        flux_model = ones(length(log_λ), n_obs)
        vars_model = ones(length(log_λ), n_obs)
        SSOF._spectra_interp_gp!(flux_model, vars_model, log_λ, data.flux, data.var .+ SSOF.SOAP_gp_var, data.log_λ_obs, gp_mean=1.)
        # init = SSOF.make_template(flux_model; use_mean=true)
        init = [mean(view(flux_model, i, :), AnalyticWeights(1 ./ view(vars_model, i, :))) for i in 1:size(flux_model, 1)]

        # optimize to get daily_flux
        # interps = SSOF.oversamp_interp_helper(data.log_λ_obs_bounds, log_λ)
        interps = SSOF.undersamp_interp_helper(data.log_λ_obs, log_λ)
        obj = optim_obj(interps, data, init)
        result = Optim.optimize(obj, init, LBFGS())
        daily_flux = result.minimizer

        # get estimates of variance
        daily_var = fit_var(daily_flux, data, interps)

        # last quantities
        airmass = mean(airmasses)
        time_nu = mean(times_nu)

        @save save_path log_λ log_λ_star daily_flux daily_var time_nu airmass
    else
        "order $desired_order not found"
    end
end
