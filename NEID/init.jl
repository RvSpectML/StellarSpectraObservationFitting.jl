## Setup
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

## Importing data with Eric's code
import StellarSpectraObservationFitting as SSOF
using EchelleInstruments, EchelleInstruments.NEID
using CSV, DataFrames, Query
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtilities/SSOFUtilities.jl")
SSOFU = SSOFUtilities

stars = ["10700", "26965", "22049", "3651", "95735", "2021/12/19", "2021/12/20", "2021/12/23"]
star_choice = SSOF.parse_args(1, Int, 5)
star = stars[star_choice]
solar = star_choice > 5
if length(ARGS) != 0; ENV["GKSwstype"] = "100" end
include("data_locs.jl")  # defines neid_data_path and neid_save_path
target_subdir = star * "/"  # needed for param.jl
if solar
	fits_target_str = "Sun"  # needed for param.jl
	df_files = make_manifest(neid_solar_data_path * target_subdir, NEID)
else
	fits_target_str = "HD " * star  # needed for param.jl
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
