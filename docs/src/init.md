# Initialization and model selection

The total SSOF model is most commonly held in a [`StellarSpectraObservationFitting.OrderModelWobble`](@ref) struct.

A good start for a SSOF model for a given dataset can be obtained with

```@docs
StellarSpectraObservationFitting.calculate_initial_model
```

which builds up the SSOF model component by component using noise-weighted [expectation maximization PCA](https://github.com/christiangil/ExpectationMaximizationPCA.jl) and find the AIC-minimum SSOF model for a given maximum amount of feature vectors.
