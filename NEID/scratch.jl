## Fixing Manifest.jl

using Pkg
Pkg.activate(".")
Pkg.precompile()
Pkg.update()

Pkg.activate("NEID")
Pkg.instantiate()

Pkg.rm("EMPCA")
Pkg.rm("StellarSpectraObservationFitting")
Pkg.rm("RvSpectMLBase")
Pkg.rm("RvSpectML")
Pkg.rm("EchelleCCFs")
Pkg.rm("EchelleInstruments")

# Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\EMPCA.jl")
# Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\StellarSpectraObservationFitting")
Pkg.develop(;path="D:\\Christian\\Documents\\GitHub\\EMPCA")
Pkg.develop(;path="C:\\Users\\Christian\\Dropbox\\GP_research\\julia\\StellarSpectraObservationFitting")
# Pkg.develop(;path="/storage/work/c/cjg66/GP_research/EMPCA/")
# Pkg.develop(;path="/storage/work/c/cjg66/GP_research/StellarSpectraObservationFittin.jl/")
Pkg.develop(;url="https://github.com/christiangil/RvSpectMLBase.jl")
Pkg.develop(;url="https://github.com/christiangil/RvSpectML.jl")
Pkg.develop(;url="https://github.com/christiangil/EchelleCCFs.jl")
Pkg.develop(;url="https://github.com/christiangil/EchelleInstruments.jl")

Pkg.instantiate()
import Pkg; Pkg.precompile()

using CSV

Pkg.rm("GLOM_RV_Example")
Pkg.rm("GPLinearODEMaker")
Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\GPLinearODEMaker.jl")
Pkg.develop(;path="C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\GLOM_RV_Example")
