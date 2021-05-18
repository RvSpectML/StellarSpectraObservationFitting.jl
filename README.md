StellarSpectraObservationFitting.jl
========

StellarSpectraObservationFitting (SSOF) is a package for a Julia package for creating data-driven linear models for the time-variable spectral features in both the observer and observed frames. SSOF outputs estimates for the radial velocities, template spectra in both the observer and barycentric frames, and scores that quantify temporal variability of time-variable telluric and stellar features.  The resulting model can be used to aid in mitigating remaining sources of correlated noise in the radial velocity time series and improving the effective precision of upcoming exoplanet surveys. 

```julia
asd
```

# Installation

This package is in rapid development so do not expect any stability yet, but the current version can be installed with the

```julia]
using Pkg
# Pkg.add(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")
# Pkg.develop(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")  # if you wanted to be able to locally edit the code easily
```
