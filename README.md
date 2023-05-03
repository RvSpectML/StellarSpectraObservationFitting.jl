StellarSpectraObservationFitting.jl (SSOF, */suf/* like soufflé)
========

<p align="center">
    <img width="400px" src="https://raw.githubusercontent.com/christiangil/StellarSpectraObservationFitting.jl/master/docs/src/assets/logo.png"/>
</p>

StellarSpectraObservationFitting (SSOF) is a Julia package for measuring radial velocities and creating data-driven models (with fast, physically-motivated Gaussian Process regularization) for the time-variable spectral features for both the telluric transmission and stellar spectrum measured by Extremely Precise Radial Velocity (EPRV) spectrographs (while accounting for the wavelength-dependent instrumental line-spread function)

# Documentation

For more details and options, see the [documentation](https://christiangil.github.io/StellarSpectraObservationFitting.jl)

# Installation

This package is in rapid development so do not expect any stability yet, but the current version can be installed with the following

```julia
using Pkg
Pkg.add("StellarSpectraObservationFitting")
# Pkg.develop(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")  # if you wanted to be able to locally edit the code easily
```

