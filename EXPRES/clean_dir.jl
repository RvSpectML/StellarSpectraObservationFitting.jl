## Importing packages
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting; SSOF = StellarSpectraObservationFitting
using Dates
## Setting up necessary variables

stars = ["10700", "26965", "34411"]
# orders_list = [1:85, 1:85, 1:85]
orders_list = [1:85, 67:69, 1:85]
include("data_locs.jl")  # defines expres_data_path and expres_save_path
# prep_str = "noreg_"
prep_str = ""
cutoff = now() - Week(1)

function clean(order::Int, star::String; delete::Bool=false)
    dir = expres_save_path*star*"/$(order)/"
    ls = readdir(dir)
    println(order)
    for file in ls
        if file != "data.jld2" && !isdir(dir * file) && (mtime(dir * file) < datetime2unix(cutoff))
            println(file)
            if delete; rm(dir * file) end
        end
    end
end

input_ind = SSOF.parse_args(1, Int, 2)
input_ind == 0 ? star_inds = (1:3) : star_inds = input_ind
for star_ind in star_inds
    star = stars[star_ind]
    orders = orders_list[star_ind]
    n_ord = length(orders)
    for i in 1:n_ord
        clean(orders[i], star; delete=false)
    end
end
