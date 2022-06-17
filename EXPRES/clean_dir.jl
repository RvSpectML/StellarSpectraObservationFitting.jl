## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using Dates
## Setting up necessary variables

stars = ["10700", "26965", "34411"]
orders_list = repeat([1:85], length(stars))
include("data_locs.jl")  # defines expres_data_path and expres_save_path
prep_str = ""
cutoff = now() - Week(1)
input_ind = SSOF.parse_args(1, Int, 0)
delete = SSOF.parse_args(2, Bool, false)

function clean(order::Int, star::String)
    dir = expres_save_path*star*"/$(order)/"
    if isdir(dir)
        ls = readdir(dir)
        println(order)
        for file in ls
            # if file != "data.jld2" && !isdir(dir * file) && (mtime(dir * file) < datetime2unix(cutoff))
            if file != "data.jld2" && (mtime(dir * file) < datetime2unix(cutoff))
                println(file)
                if delete; rm(dir * file) end
            end
        end
    else
        println("couldn't find " * dir)
    end
end

input_ind == 0 ? star_inds = (1:length(stars)) : star_inds = input_ind
for star_ind in star_inds
    for order in orders_list[star_ind]
        clean(order, stars[star_ind])
    end
end
