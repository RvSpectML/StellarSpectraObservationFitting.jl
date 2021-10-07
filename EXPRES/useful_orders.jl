## Importing packages
using Pkg
Pkg.activate("EXPRES")

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting

## Setting up necessary variables

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 1)]
interactive = length(ARGS) == 0
include("data_locs.jl")  # defines expres_data_path and expres_save_path

using Distributed
function sendto(workers::Union{T,Vector{T}}; args...) where {T<:Integer}
    for worker in workers
        for (var_name, var_value) in args
            @spawnat(worker, Core.eval(Main, Expr(:(=), var_name, var_value)))
        end
    end
end
addprocs(length(Sys.cpu_info()) - 2)
@everywhere using Pkg; @everywhere Pkg.activate("EXPRES");
@everywhere using StellarSpectraObservationFitting; @everywhere SSOF = StellarSpectraObservationFitting
@everywhere using JLD2
@everywhere using Statistics
datapath = expres_save_path * star
sendto(workers(), datapath=datapath, star=star)
@everywhere function f(desired_order::Int)
    @load datapath * "/$(desired_order)/data.jld2" n_obs data times_nu airmasses
    model = SSOF.OrderModel(data, "EXPRES", desired_order, star; n_comp_tel=8, n_comp_star=8)
    SSOF.initialize!(model, data; use_gp=true)
    o = SSOF.Output(model, data)
    return stdm(data.flux - SSOF.total_model(o.tel, o.star, o.rv), 0)
end
ords = 1:85
res = pmap(x->f(x), ords, batch_size=Int(floor(length(ords) / (nworkers() + 1)) + 1))

@save "useful_order_res_$star.jld2" res

using Plots
plot(res; yaxis=:log)
