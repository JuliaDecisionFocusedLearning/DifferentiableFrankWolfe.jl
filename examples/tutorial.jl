# # Tutorial

# Necessary imports

using DifferentiableFrankWolfe: DiffFW, simplex_projection
using ForwardDiff: ForwardDiff
using FrankWolfe: ProbabilitySimplexOracle
using ProximalOperators: ProximalOperators
using Test: @test
using Zygote: Zygote

# Constructing the wrapper

f(x, θ) = 0.5 * sum(abs2, x - θ)  # minimizing the squared distance...
f_grad1(x, θ) = x - θ
lmo = ProbabilitySimplexOracle(1.0)  # ... to the probability simplex
dfw = DiffFW(f, f_grad1, lmo);  # ... is equivalent to a simplex projection if we're not already in it

# Calling the wrapper

θ = float.(1:5)

#-

frank_wolfe_kwargs = (; max_iteration=100, epsilon=1e-4)
y = dfw(θ, frank_wolfe_kwargs)

#- Comparing with the ground truth

true_simplex_projection(x) = ProximalOperators.prox(ProximalOperators.IndSimplex(1.0), x)[1]

#-

y_true = true_simplex_projection(θ)
@test Vector(y) ≈ Vector(y_true) atol = 1e-3

# Differentiating the wrapper

J1 = Zygote.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs), θ)[1]
J1_true = Zygote.jacobian(true_simplex_projection, θ)[1]
@test J1 ≈ J1_true atol = 1e-3

#-

J2 = ForwardDiff.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs), θ)
J2_true = ForwardDiff.jacobian(true_simplex_projection, θ)
@test J2 ≈ J2_true atol = 1e-3
