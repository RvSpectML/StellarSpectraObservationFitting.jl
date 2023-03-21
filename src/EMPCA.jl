## EMPCA alias 

import ExpectationMaximizationPCA as EMPCA
using LinearAlgebra

"""
	EMPCA!(lm, data_tmp, weights; kwargs...)

Perform Expectation maximization PCA on `data_temp`. See https://github.com/christiangil/ExpectationMaximizationPCA.jl
"""
EMPCA.EMPCA!(lm::FullLinearModel, data_tmp::AbstractMatrix, weights::AbstractMatrix; kwargs...) =
	EMPCA.EMPCA!(lm.M, lm.s, lm.μ, data_tmp, weights; use_log=log_lm(lm), kwargs...)
