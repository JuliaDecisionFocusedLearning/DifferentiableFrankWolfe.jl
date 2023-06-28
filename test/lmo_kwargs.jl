using DifferentiableFrankWolfe
using FrankWolfe
using Test

minimizer(θ; a=0) = θ + a
maximizer(θ; a=0) = θ + a

lmokw = LinearMinimizationOracleWithKwargs(minimizer)
@test FrankWolfe.compute_extreme_point(lmokw, 2) == 2 + 0

lmokw = LinearMaximizationOracleWithKwargs(maximizer)
@test FrankWolfe.compute_extreme_point(lmokw, 2) == -2 + 0

lmokw = LinearMinimizationOracleWithKwargs(minimizer, (; a=1))
@test FrankWolfe.compute_extreme_point(lmokw, 2) == 2 + 1

lmokw = LinearMaximizationOracleWithKwargs(maximizer, (; a=1))
@test FrankWolfe.compute_extreme_point(lmokw, 2) == -2 + 1
