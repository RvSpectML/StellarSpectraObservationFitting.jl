# Data preparation

SSOF models are meant for clean, high-resolution, continuum-normalized spectra. These spectra should be stored in the [`StellarSpectraObservationFitting.GenericData`](@ref) and [`StellarSpectraObservationFitting.LSFData`](@ref) objects, which are used to ensure that all of the necessary information exists to optimize a SSOF model.

```@docs
StellarSpectraObservationFitting.GenericData
```

```@docs
StellarSpectraObservationFitting.LSFData
```

The functions for creating these objects from observation .fits files are currently outside of SSOF proper to keep unnecessary dependencies down, but you can see the reformat_spectra function in [SSOFUtilities/init.jl](https://github.com/christiangil/StellarSpectraObservationFitting.jl/blob/master/SSOFUtilities/init.jl) (which flags low SNR observations and those with weird wavelength calibration as well as ) and [NEID/init.jl](https://github.com/christiangil/StellarSpectraObservationFitting.jl/blob/master/NEID/init.jl) for a script using it.

Once the data is collected, we recommend running [`StellarSpectraObservationFitting.process!`](@ref) to perform some data preprocessing.

```@docs
StellarSpectraObservationFitting.process!
```