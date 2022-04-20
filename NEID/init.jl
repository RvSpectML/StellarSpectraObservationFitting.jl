## Setup
using Pkg
Pkg.activate("NEID")
Pkg.instantiate()

## Importing data with Eric's code
import StellarSpectraObservationFitting as SSOF
using EchelleInstruments, EchelleInstruments.NEID
using CSV, DataFrames, Query
SSOF_path = dirname(dirname(pathof(SSOF)))
include(SSOF_path * "/SSOFUtliities/SSOFUtilities.jl")
SSOFU = SSOFUtilities

# stars = ["2021/01/30", "2021/02/04", "2021/02/06", "2021/02/08", "2021/02/11", "2021/02/12", "2021/02/14", "2021/02/18", "2021/02/20", "2021/02/21", "2021/02/22", "2021/02/24", "2021/02/25", "2021/03/01", "2021/03/02", "2021/03/05", "2021/03/06", "2021/03/10", "2021/03/14", "2021/03/17", "2021/03/20", "2021/03/22", "2021/03/27", "2021/03/28", "2021/03/30", "2021/03/31", "2021/04/02", "2021/04/03", "2021/04/04", "2021/04/13", "2021/04/17", "2021/04/18", "2021/04/19", "2021/04/21", "2021/04/23", "2021/04/24", "2021/04/29", "2021/04/30", "2021/05/03", "2021/05/04", "2021/05/06", "2021/05/07", "2021/05/10", "2021/05/11", "2021/05/12", "2021/05/14", "2021/05/16", "2021/05/17", "2021/05/19", "2021/05/21", "2021/05/22", "2021/05/27", "2021/05/28", "2021/05/29", "2021/05/30", "2021/05/31", "2021/06/01", "2021/06/02", "2021/06/03", "2021/06/05", "2021/06/06", "2021/06/11", "2021/06/12", "2021/06/13", "2021/06/14", "2021/06/19", "2021/06/20", "2021/06/21", "2021/06/25", "2021/06/27", "2021/07/10", "2021/07/11", "2021/07/13", "2021/07/28", "2021/08/02", "2021/09/20", "2021/09/21", "2021/09/28", "2021/10/28", "2021/10/29", "2021/10/30", "2021/10/31", "2021/11/03", "2021/11/04", "2021/11/06", "2021/11/08", "2021/11/10", "2021/11/11", "2021/11/12", "2021/11/13", "2021/11/14", "2021/11/15", "2021/11/18", "2021/11/26", "2021/11/27", "2021/11/28", "2021/11/29", "2021/12/03", "10700"]
stars = ["10700", "9407", "2021/12/19"]
star_choice = SSOF.parse_args(1, Int, 1)
star = stars[star_choice]
solar = star_choice > 2
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

SSOFU.reformat_spectra(
	df_files_use,
	neid_save_path * target_subdir,
	NEID,
	min_order(NEID2D()):max_order(NEID2D()))
