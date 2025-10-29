@testitem "Constructor" begin
    using FrankWolfe

    f(x, θ) = 0.5 * sum(abs2, x - θ)
    f_grad1(x, θ) = x - θ
    lmo = FrankWolfe.UnitSimplexLMO(1.0)

    dfw1 = DiffFW(f, f_grad1, lmo)
    @test dfw1.implicit.linear_solver != \

    implicit_kwargs = (; linear_solver = \)
    dfw2 = DiffFW(f, f_grad1, lmo; implicit_kwargs)
    @test dfw2.implicit.linear_solver == \
end

@testitem "Projection" setup = [Setup] begin
    import DifferentiableFrankWolfe as DFW
    using ForwardDiff
    using StableRNGs
    using Test
    using Zygote

    for n in (2, 5, 10), scaling in (0.1, 1, 10), _ in 1:10
        x = scaling .* rand(StableRNG(n), n)
        @test DFW.simplex_projection(x) ≈ true_simplex_projection(x)
        J = Zygote.jacobian(DFW.simplex_projection, x)[1]
        J_true = ForwardDiff.jacobian(true_simplex_projection, x)
        @test J ≈ J_true
    end
end

@testitem "DiffFW" setup = [Setup] begin
    using ForwardDiff
    using FrankWolfe
    using StableRNGs
    using Test
    using Zygote

    fwkw = (; max_iteration = 1000, epsilon = 1.0e-4, line_search = FrankWolfe.Agnostic())

    @testset "Simplex projection" begin
        lmo = FrankWolfe.ProbabilitySimplexLMO(1.0)
        dfw = DiffFWProjection(lmo)

        for n in (2, 5, 10), scaling in (0.1, 1, 10)
            θ = scaling .* rand(StableRNG(n), n)  # outside of the simplex a.s.
            x0 = ones(n) ./ n
            @test dfw(θ, x0; fwkw...) ≈ true_simplex_projection(θ) rtol = 1.0e-1
            J = Zygote.jacobian(_θ -> dfw(_θ, x0; fwkw...), θ)[1]
            J_true = ForwardDiff.jacobian(true_simplex_projection, θ)
            @test J ≈ J_true rtol = 1.0e-2
        end
    end

    @testset "Ball projection" begin
        lmo = FrankWolfe.LpNormLMO{2}(1.0)
        dfw = DiffFWProjection(lmo)

        θ = float.(1:5)  # outside of the ball, projected to single atom, no derivative
        x0 = zero(θ)
        @test dfw(θ, x0; fwkw...) ≈ true_ball_projection(θ) rtol = 1.0e-3
        J = Zygote.jacobian(_θ -> dfw(_θ, x0; fwkw...), θ)[1]
        @test all(J .≈ 0)
    end

    @testset "Different sizes" begin
        using FrankWolfe
        using DifferentiableFrankWolfe
        using Zygote
        using Test

        f(x, θ) = 0.5 * sum(abs2, x .- sqrt.(vcat(θ, sum(θ))))
        f_grad1(x, θ) = x .^ 2 .- vcat(θ, sum(θ))
        lmo = FrankWolfe.ScaledBoundLInfNormBall(zeros(3), ones(3))
        dfw = DiffFW(f, f_grad1, lmo; step_size = 1)
        θ = [0.3, 0.2]
        x0 = [0.7, 0.5, 0.9]
        fwkw = (; max_iteration = 1000, epsilon = 1.0e-4)
        @test dfw(θ, x0; fwkw...) ≈ sqrt.(vcat(θ, sum(θ))) rtol = 1.0e-4
        J = Zygote.jacobian(_θ -> dfw(_θ, x0; fwkw...), θ)[1]
        J_ref = Zygote.jacobian(_θ -> sqrt.(vcat(_θ, sum(_θ))), θ)[1]
        @test J ≈ J_ref rtol = 1.0e-4
    end
end

@testitem "Tutorial" begin
    include(joinpath(@__DIR__, "..", "examples", "tutorial.jl"))
end
