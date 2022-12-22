## Setup
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

## Importing data with Eric's code
import StellarSpectraObservationFitting as SSOF
using EchelleInstruments, EchelleInstruments.NEID
using CSV, DataFrames, Query
using FITSIO
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities

stars = ["26965", "3651", "Barnard"]
fits_target_strs = [["HD 26965"], ["HD 3651"], ["GJ699", "TIC 325554331"]]
star_choice = SSOF.parse_args(1, Int, 3)
star = stars[star_choice]
solar = star_choice > 3
if length(ARGS) != 0; ENV["GKSwstype"] = "100" end
include("data_locs.jl")  # defines neid_data_path and neid_save_path
target_subdir = star * "/"  # needed for param.jl
if solar
	fits_target_str = ["Sun"]  # needed for param.jl
	df_files = make_manifest(neid_solar_data_path * target_subdir, NEID)
else
	fits_target_str = fits_target_strs[star_choice]
	df_files = make_manifest(neid_data_path * target_subdir, NEID)
end
include("param.jl")  # filters df_files -> df_files_use

include("lsf.jl")  # defines NEIDLSF.NEID_lsf()
SSOFU.reformat_spectra(
	df_files_use,
	neid_save_path * target_subdir,
	NEID,
	min_order(NEID2D()):118,
	star;
	lsf_f = NEIDLSF.neid_lsf,
	interactive=length(ARGS)==0,
	min_snr=5)
SSOFU.neid_extras(df_files_use, neid_save_path * target_subdir)
