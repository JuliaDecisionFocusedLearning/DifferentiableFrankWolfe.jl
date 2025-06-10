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

@testitem "Tutorial" begin
    include(joinpath(@__DIR__, "..", "examples", "tutorial.jl"))
end
