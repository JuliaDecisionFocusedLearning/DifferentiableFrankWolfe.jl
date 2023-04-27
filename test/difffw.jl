using ChainRulesCore
using FrankWolfe
using DifferentiableFrankWolfe: DiffFW, simplex_projection
using Random
using Statistics
using Test
using Zygote

Random.seed!(63)

d = 100
θ = rand(d);
dy = rand(d);
rc = Zygote.ZygoteRuleConfig()

f(x, θ) = sum(abs2, x - θ) / 2
f_grad1(x, θ) = x - θ
lmo = FrankWolfe.UnitSimplexOracle(1.0)
frank_wolfe_kwargs = (max_iteration=500, epsilon=1e-5)
dfw = DiffFW(f, f_grad1, lmo)

_, pullback_simplex_projection = rrule_via_ad(rc, simplex_projection, θ);
_, pullback_dfw = rrule_via_ad(rc, dfw, θ; frank_wolfe_kwargs=frank_wolfe_kwargs);

@test mean(abs, dfw(θ; frank_wolfe_kwargs=frank_wolfe_kwargs) - simplex_projection(θ)) <
    1e-3
@test mean(abs, pullback_dfw(dy)[2] - pullback_simplex_projection(dy)[2]) < 1e-3
