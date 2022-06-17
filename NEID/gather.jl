## Importing packages
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

using JLD2
import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities

## Setting up necessary variables

input_ind = SSOF.parse_args(1, Int, 0)
dpca = SSOF.parse_args(2, Bool, true)

stars = ["10700", "26965", "9407", "185144", "22049", "2021/12/19", "2021/12/20", "2021/12/23"]
input_ind == 0 ? star_inds = (1:length(stars)) : star_inds = input_ind:input_ind
orders_list = repeat([4:122], length(stars))
include("data_locs.jl")  # defines neid_data_path and neid_save_path
if dpca
    prep_str = ""
else
    prep_str = "wobble/"
end

SSOFU.retrieve_all_rvs(
    [neid_save_path*star*"/60/data.jld2" for star in stars[star_inds]],
    [[neid_save_path*stars[i]*"/$order/$(prep_str)results.jld2" for order in orders_list[i]] for i in star_inds],
    ["jld2/neid_" * replace("$(prep_str)$(star)_rvs.jld2", "/" => "_") for star in stars[star_inds]]
    )
