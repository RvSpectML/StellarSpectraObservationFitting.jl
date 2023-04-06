# Error estimation

The RVs found by SSOF are useless if we don't know how confident we should be in them.
We have implemented 2 methods for estimating the uncertainties on the RVs and model scores based on the photon uncertainties in the original data.

For getting quick estimates of the uncertainties, we can look at the local curvature of the likelihood space. 
If one assumes that loss is approximately a Gaussian log-likelihood, then the covariance matrix, $\Sigma_{\beta_M}$ can be approximated as 

$$\Sigma_{\beta_M} \approx (-H(\beta_M))^{-1}$$

where 

$$H(\beta_M)_{i,j}=\dfrac{\delta^2 \ell(\beta_M)}{\delta \beta_{M,i} \delta \beta_{M,j}}$$

is the Hessian matrix and $\ell(\beta_M)$ is the $\approx$ Gaussian log-likelihood (which in our case is $\dfrac{-\mathcal{L}}{2}$).
The variance of each model parameter can be further approximated assuming that the off-diagonal entries of $H(\beta_M)$ are zero (i.e. assuming any $\beta_{M,i}$ is uncorrelated with $\beta_{M,j}$)

$$\dfrac{1}{\sigma_{\beta_{M,i}}^2} \approx -\dfrac{\delta^2 \ell(\beta_M)}{\delta \beta_{M,i}^2}$$

We effectively approximate $\dfrac{\delta^2 \ell(\beta_M)}{\delta \beta_{M,i}^2}$ with finite differences.
This is made available the user with

```@docs
StellarSpectraObservationFitting.estimate_σ_curvature
```

This method is very fast and recommended when performing repeated, iterative analyses (e.g. during data exploration or survey simulation).

Another method available in SSOF for estimating errors is via bootstrap resampling.
In this method, we repeatedly refit the model to the data after adding white noise to each pixel at the reported variance levels.
An estimate for the covariance of $\beta_M$ can then be found by looking at the distribution of the proposed $\beta_M$ after the repeated refittings.
These estimates for the uncertainties tend to be $\sim 1.1-1.5$x higher than the loss space curvature based estimates (likely due to the ignored off-diagonal terms in $H(\beta_M)$).
This method is slower but gives a better overall estimate for the uncertainties (and covariances if desired) and is recommended when finalizing analysis results.

```@docs
StellarSpectraObservationFitting.estimate_σ_bootstrap
```

