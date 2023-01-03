StellarSpectraObservationFitting.jl
========

StellarSpectraObservationFitting (SSOF) is a Julia package for creating data-driven models (with fast, physically-motivated Gaussian Process regularization) for the time-variable spectral features for both the telluric transmission and stellar spectrum measured by Extremely Precise Radial Velcotiy (EPRV) spectrographs. SSOF outputs estimates for the radial velocities, template spectra in both the observer and barycentric frames, and scores and feature vectors that quantify temporal variability of time-variable telluric and stellar features, while accounting for the wavelength-dependent instrumental line-spread function (LSF).

# Installation

This package is in rapid development so do not expect any stability yet, but the current version can be installed with the following

```julia]
using Pkg
Pkg.add(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")
# Pkg.develop(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")  # if you wanted to be able to locally edit the code easily
```
