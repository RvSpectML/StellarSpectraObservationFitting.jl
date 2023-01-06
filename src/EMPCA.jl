## EMPCA alias 

import ExpectationMaximizationPCA as EMPCA
using LinearAlgebra

EMPCA.EMPCA!(lm::FullLinearModel, data_tmp::AbstractMatrix, weights::AbstractMatrix; kwargs...) =
	EMPCA.EMPCA!(lm.M, lm.s, lm.μ, data_tmp, weights; use_log=log_lm(lm), kwargs...)
