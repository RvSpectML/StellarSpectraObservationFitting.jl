## Importing packages
using Pkg
Pkg.activate("NEID")

import StellarSpectraObservationFitting as SSOF
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities
using JLD2

## Setting up necessary variables

stars = ["10700", "26965", "22049", "3651", "95735", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 2)
star = stars[star_choice]
include("../NEID/lsf.jl")  # defines NEIDLSF.NEID_lsf()
include("data_locs.jl")  # defines neid_data_path and neid_save_path
nlsf = NEIDLSF
npix = 30
lsf_orders = [i for i in 1:length(nlsf.no_lsf_orders) if !nlsf.no_lsf_orders[i]]
spacing = Array{Float64}(undef, length(lsf_orders))
# for desired_order in lsf_orders
for i in 1:length(lsf_orders)
	desired_order = lsf_orders[i]
	base_path = neid_save_path * star * "/$(desired_order)/"
	data_path = base_path * "data.jld2"
	@load data_path data
	j = Int(round(size(data.log_λ_obs,1)/2))
	spacing[i] = mean(data.log_λ_obs[j+1, :] - data.log_λ_obs[j-1, :]) / 2
end
spacing
@save "order_pixel_spacing.jld2" spacing lsf_orders
