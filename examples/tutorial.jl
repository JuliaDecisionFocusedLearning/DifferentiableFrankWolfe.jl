# # Tutorial

# Necessary imports

using DifferentiableFrankWolfe: DiffFW, simplex_projection
using ForwardDiff: ForwardDiff
using FrankWolfe: ProbabilitySimplexLMO
using ProximalOperators: ProximalOperators
using Test: @test
using Zygote: Zygote

# Constructing the wrapper

f(x, θ) = 0.5 * sum(abs2, x - θ)  # minimizing the squared distance...
f_grad1(x, θ) = x - θ
lmo = ProbabilitySimplexLMO(1.0)  # ... to the probability simplex
dfw = DiffFW(f, f_grad1, lmo);  # ... is equivalent to a simplex projection if we're not already in it

# Calling the wrapper

x0 = ones(3) ./ 3
θ = [1.0, 1.5, 0.2]

#-

frank_wolfe_kwargs = (; max_iteration = 100, epsilon = 1.0e-4)
y = dfw(θ, x0; frank_wolfe_kwargs...)

#- Comparing with the ground truth

true_simplex_projection(x) = ProximalOperators.prox(ProximalOperators.IndSimplex(1.0), x)[1]

#-

y_true = true_simplex_projection(θ)
@test Vector(y) ≈ Vector(y_true) atol = 1.0e-3

# Differentiating the wrapper

#-

J_true = ForwardDiff.jacobian(true_simplex_projection, θ)

#-

J1 = Zygote.jacobian(_θ -> dfw(_θ, x0; frank_wolfe_kwargs...), θ)[1]
@test J1 ≈ J_true atol = 1.0e-3

#-

J2 = ForwardDiff.jacobian(_θ -> dfw(_θ, x0; frank_wolfe_kwargs...), θ)
@test J2 ≈ J_true atol = 1.0e-3
