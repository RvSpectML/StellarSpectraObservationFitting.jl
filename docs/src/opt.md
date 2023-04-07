# Optimization

Before optimization, the SSOF problem (with a SSOF model and the [`StellarSpectraObservationFitting.LSFData`](@ref) it's being fit with) is organized into a work space (like [`StellarSpectraObservationFitting.TotalWorkspace`](@ref)) which includes a suitable chi-squared loss function and its gradient

$$\mathcal{L}(\beta_M) = \sum_{n=1}^N (Y_{D,n} - Y_{M,n})^T \Sigma_n^{-1} (Y_{D,n} - Y_{M,n}) + \textrm{constant}$$

This object can be passed to a function like [`StellarSpectraObservationFitting.improve_model!`](@ref) to optimize the SSOF model on the data.

```@docs
StellarSpectraObservationFitting.improve_model!
```

