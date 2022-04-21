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

input_ind = SSOF.parse_args(1, Int, 3)

stars = ["10700", "9407", "2021/12/19", "2021/12/20", "2021/12/23"]
input_ind == 0 ? star_inds = (1:length(stars)) : star_inds = input_ind:input_ind
orders_list = [4:122, 4:122, 4:122, 4:122, 4:122]
include("data_locs.jl")  # defines neid_data_path and neid_save_path
# prep_str = "noreg_"
prep_str = ""

SSOFU.retrieve_all_rvs(
    [neid_save_path*star*"/60/data.jld2" for star in stars[star_inds]],
    [[neid_save_path*stars[i]*"/$order/$(prep_str)results.jld2" for order in orders_list[i]] for i in star_inds],
    [replace("neid_$(prep_str)$(star)_rvs.jld2", "/" => "_") for star in stars[star_inds]]
    )
