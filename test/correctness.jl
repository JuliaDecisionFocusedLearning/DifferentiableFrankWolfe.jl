@testitem "Constructor" begin
    using FrankWolfe

    f(x, θ) = 0.5 * sum(abs2, x - θ)
    f_grad1(x, θ) = x - θ
    lmo = FrankWolfe.UnitSimplexOracle(1.0)

    dfw1 = DiffFW(f, f_grad1, lmo)
    @test dfw1.implicit.linear_solver != \

    implicit_kwargs = (; linear_solver=\)
    dfw2 = DiffFW(f, f_grad1, lmo; implicit_kwargs)
    @test dfw2.implicit.linear_solver == \
end

@testitem "Projection" begin
    using ProximalOperators
    using DifferentiableFrankWolfe: simplex_projection
    using Test
    using Zygote

    for n in (2, 5, 10, 50)
        for _ in 1:10
            x = rand(n)
            @test simplex_projection(x) ≈ prox(IndSimplex(1.0), x)[1]
            @test Zygote.jacobian(simplex_projection, x)[1] ≈
                ForwardDiff.jacobian(simplex_projection, x)
        end
    end
end

@testitem "Tutorial" begin
    include(joinpath(@__DIR__, "..", "examples", "tutorial.jl"))
end
