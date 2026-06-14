# Optimization

Before optimization, the SSOF problem (with a SSOF model and the [`StellarSpectraObservationFitting.LSFData`](@ref) it's being fit with) is organized into a work space (like [`StellarSpectraObservationFitting.TotalWorkspace`](@ref)) which includes a suitable chi-squared loss function and its gradient

$$\mathcal{L}(\beta_M) = \sum_{n=1}^N (Y_{D,n} - Y_{M,n})^T \Sigma_n^{-1} (Y_{D,n} - Y_{M,n}) + \textrm{constant}$$

This object can be passed to a function like [`StellarSpectraObservationFitting.improve_model!`](@ref) to optimize the SSOF model on the data.

```@docs
StellarSpectraObservationFitting.improve_model!
```

## DPCA loss semantics (migration note)

For `OrderModelDPCA` models, `loss_funcs_total` recomputes the Doppler basis inline from `om.star.lm.μ` on every call.
This means the gradient flows ∂loss/∂μ → ∂doppler\_basis, so optimizing μ automatically updates the Doppler component.
The pre-port Nabla implementation treated the Doppler basis as a constant (no gradient through μ → basis).
The Wobble path (`OrderModelWobble`) is unaffected by this change.

