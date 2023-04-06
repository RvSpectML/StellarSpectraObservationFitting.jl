# Regularization

Many different regularizations can be added to the SSOF loss function to prevent overfitting and encourage model sparsity and smoothness for the model templates and feature vectors. Namely the classic L1 and L2 norms, as well as a fast 𝒪(n) GP term.

$$\mathcal{L}_{R}(\beta_M, \beta_R) = \mathcal{L}(\beta_M) + a_1 \ell_{\textrm{LSF}}(\xi_\oplus, \mu_\oplus - 1) + a_2 \ell_{\texttt{SOAP}}(\xi_\star, \mu_\star - 1) + a_3||\mu_\oplus||_2^2 + a_4||\mu_\star||_2^2 + a_5||\mu_\oplus||_1^1 + a_6||\mu_\star||_1^1$$

$$+ a_7 \sum_i^{K_\oplus} \ell_{\textrm{LSF}}(\xi_\oplus, W_{\oplus,i}) + a_8 \sum_i^{K_\star} \ell_{\texttt{SOAP}}(\xi_\star, W_{\star,i}) + a_9||W_\oplus||_2^2 + a_{10}||W_\star||_2^2 + a_{11}||W_\oplus||_1^1 + a_{12}||W_\star||_1^1 + ||S_\oplus||_2^2 + ||S_\star||_2^2$$

while the regularization coefficients have default values, you can find the optimal set for the given SSOF model and dataset with cross validation using [`StellarSpectraObservationFitting.improve_regularization!`](@ref)

```@docs
StellarSpectraObservationFitting.improve_regularization!
```

The math behind the Kalman-filtering based methods for the 𝒪(n) GP inference can be found in the appendix of the SSOF paper (submitted) while the actual code is in [src/prior_gp_functions.jl](https://github.com/christiangil/StellarSpectraObservationFitting.jl/blob/master/src/prior_gp_functions.jl)