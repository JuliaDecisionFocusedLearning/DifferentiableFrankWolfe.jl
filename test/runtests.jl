using TestItems
using TestItemRunner

@testsnippet Setup begin
    using DifferentiableFrankWolfe
    using ProximalOperators

    true_simplex_projection(x) = prox(IndSimplex(1.0), x)[1]
    true_ball_projection(x) = prox(IndBallL2(1.0), x)[1]

    function DiffFWProjection(lmo)
        f(x, θ) = 0.5 * sum(abs2, x - θ)
        f_grad1(x, θ) = x - θ
        return DiffFW(f, f_grad1, lmo)
    end
end

@run_package_tests
