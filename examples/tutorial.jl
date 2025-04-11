# # Tutorial

# Necessary imports

using DifferentiableFrankWolfe: DiffFW, simplex_projection
using ForwardDiff: ForwardDiff
using FrankWolfe: UnitSimplexOracle
using Test: @test
using Zygote: Zygote

# Constructing the wrapper

f(x, θ) = 0.5 * sum(abs2, x - θ)  # minimizing the squared distance...
f_grad1(x, θ) = x - θ
lmo = UnitSimplexOracle(1.0)  # ... to the probability simplex
dfw = DiffFW(f, f_grad1, lmo);  # ... is equivalent to a simplex projection

# Calling the wrapper

θ = rand(10)

#-

frank_wolfe_kwargs = (; max_iteration=100, epsilon=1e-4)
y, stats = dfw(θ, frank_wolfe_kwargs)
y

#-

y_true = simplex_projection(θ)
@test Vector(y) ≈ Vector(y_true) atol = 1e-3

# Differentiating the wrapper

J1 = Zygote.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs)[1], θ)[1]
J1_true = Zygote.jacobian(simplex_projection, θ)[1]
@test J1 ≈ J1_true atol = 1e-3

#-

J2 = ForwardDiff.jacobian(_θ -> dfw(_θ, frank_wolfe_kwargs)[1], θ)
J2_true = ForwardDiff.jacobian(simplex_projection, θ)
@test J2 ≈ J2_true atol = 1e-3
