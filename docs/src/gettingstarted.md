# Getting started

## Installation

The most current, tagged version of [StellarSpectraObservationFitting.jl](https://github.com/christiangil/StellarSpectraObservationFitting.jl) will soon be able to be easily installed using Julia's Pkg

```julia
Pkg.add("StellarSpectraObservationFitting")
```

For now, use

```julia
using Pkg
Pkg.add(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")
# Pkg.develop(;url = "https://github.com/christiangil/StellarSpectraObservationFitting.jl")  # if you wanted to be able to locally edit the code easily
```

## Example

An example notebook can be found [here](https://github.com/christiangil/StellarSpectraObservationFitting.jl/blob/master/examples/example.ipynb)

## Getting Help

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system. For example here's how to get
additional information on [`StellarSpectraObservationFitting.calculate_initial_model`](@ref) within Julia's REPL:

```julia
?StellarSpectraObservationFitting.calculate_initial_model
```

If you encounter a bug or would like to participate in the
development of this package come find us on Github.

- [christiangil/StellarSpectraObservationFitting.jl](https://github.com/christiangil/StellarSpectraObservationFitting.jl)