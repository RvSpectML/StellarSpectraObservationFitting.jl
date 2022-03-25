## Setup
using Pkg
Pkg.activate("EXPRES")
Pkg.instantiate()

## Importing data with Eric's code
import StellarSpectraObservationFitting as SSOF
using EchelleInstruments, EchelleInstruments.EXPRES
using CSV, DataFrames, Query
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities

stars = ["10700", "26965", "34411"]
star = stars[SSOF.parse_args(1, Int, 1)]
include("data_locs.jl")  # defines expres_data_path and expres_save_path
target_subdir = star * "/"  # needed for param.jl
fits_target_str = star  # needed for param.jl
include("lsf.jl")  # defines EXPRES_lsf()

df_files = make_manifest(expres_data_path * target_subdir, EXPRES)
include("param.jl")  # filters df_files -> df_files_use

SSOFU.reformat_spectra(
	df_files_use,
	expres_save_path,
	EXPRES,
	min_order(EXPRES2D()):max_order(EXPRES2D());
	lsf_f = EXPRES_lsf)
