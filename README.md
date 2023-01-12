StellarSpectraObservationFitting.jl
========
StellarSpectraObservationFitting (SSOF) is a Julia package for creating data-driven models (with fast, physically-motivated Gaussian Process regularization) for the time-variable spectral features for both the telluric transmission and stellar spectrum measured by Extremely Precise Radial Velcotiy (EPRV) spectrographs. 

![SSOF model](/docs/ssof_model.PNG "SSOF model")

SSOF outputs estimates for the radial velocities, template spectra in both the observer and barycentric frames, and scores and feature vectors that quantify temporal variability of time-variable telluric and stellar features, while accounting for the wavelength-dependent instrumental line-spread function (LSF).

# Documentation

For more details and options, see the [documentation](https://christiangil.github.io/StellarSpectraObservationFitting.jl) (available soon)

# Installation

This package is in rapid development so do not expect any stability yet, but the current version can be installed with the following

```julia
using Pkg
Pkg.add(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")
# Pkg.develop(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")  # if you wanted to be able to locally edit the code easily
```

