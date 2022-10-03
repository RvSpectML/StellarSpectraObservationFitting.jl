## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
using Dates
## Setting up necessary variables

stars = ["10700", "26965", "22049", "3651", "95735", "2021/12/19", "2021/12/20", "2021/12/23"]
orders_list = repeat([7:118], length(stars))
include("data_locs.jl")  # defines expres_data_path and expres_save_path
cutoff = now() - Week(1)
# cutoff = DateTime(2022,8,19)
input_ind = SSOF.parse_args(1, Int, 2)
how_deep = SSOF.parse_args(2, Int, 1)
delete = SSOF.parse_args(3, Bool, false)

mtime_r(fn::String) = isdir(fn) ?
    maximum(mtime_r.(fn .* readdir(fn))) :
    mtime(fn)

function clean(dir::String, level::Int)
    @assert isdir(dir) "couldn't find " * dir
    if level > 0
        for file in readdir(dir)
            if isdir(dir * file)
                clean(dir * file * "/", level - 1)
            end
        end
    else
        for file in readdir(dir)
            if file != "data.jld2" && (mtime_r(dir * file) < datetime2unix(cutoff))
                println(dir*file)
                if delete; rm(dir * file; recursive=true) end
            end
        end
    end
end

input_ind == 0 ? star_inds = (1:length(stars)) : star_inds = input_ind
for star_ind in star_inds
    for order in orders_list[star_ind]
        clean(neid_save_path*stars[star_ind]*"/$(order)/", how_deep)
    end
end
