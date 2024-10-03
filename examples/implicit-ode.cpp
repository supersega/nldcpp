#include <iostream>
#include <nld/math.hpp>
#include <vector>

#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

auto ode_from_article(const nld::vector_xd &y, double t,
                      const nld::vector_xd &ydot) -> nld::vector_xd {
    nld::vector_xd dy(1);
    dy(0) = 1.0 / 16.0 * (sin(t * t * ydot(0)) - sin(exp(y(0)))) + 1.0 / t;
    // dy(0) = 1.0 / t;
    return dy;
};

auto beam_ode_explicit(const nld::vector_xd &y, double t) -> nld::vector_xd {
    nld::vector_xd dy(2);

    auto alpha_1 = 1.182;
    auto alpha_2 = 5.5;
    auto xi = 0.002;
    auto q = 0.014;
    auto Omega = 1.08;

    dy(0) = y(1);
    dy(1) = -2.0 * xi * y(1) - 2.0 * alpha_1 * y(0) * y(1) * y(1) -
            (1 - 2.0 * q * cos(2.0 * Omega * t)) * y(0) -
            alpha_2 * y(0) * y(0) * y(0);
    dy(1) /= (1.0 + 2.0 * alpha_1 * y(0) * y(0));

    return dy;
}

auto solve_beam_ode_explicit() {
    nld::vector_xd initial = nld::vector_xd::Zero(2);
    initial(0) = 0.01;
    initial(1) = 0.02;

    auto parameters =
        nld::math::constant_step_parameters{0.0, 500.0, 500 * 300};
    auto solution = nld::math::runge_kutta_4::solution(beam_ode_explicit,
                                                       parameters, initial);

    std::vector<double> y(solution.col(0).data(),
                          solution.col(0).data() + solution.rows());
    std::vector<double> x(solution.col(2).data(),
                          solution.col(2).data() + solution.rows());

    plt::plot(x, y);
}

auto beam_ode_implicit(const nld::vector_xd &y, double t,
                       const nld::vector_xd &ydot) -> nld::vector_xd {
    nld::vector_xd dy(2);

    auto alpha_1 = 1.182;
    auto alpha_2 = 5.5;
    auto xi = 0.002;
    auto q = 0.014;
    auto Omega = 1.08;

    dy(0) = y(1);
    dy(1) = -2.0 * xi * y(1) - 2.0 * alpha_1 * y(0) * y(1) * y(1) -
            (1 - 2.0 * q * cos(2.0 * Omega * t)) * y(0) -
            alpha_2 * y(0) * y(0) * y(0) -
            2.0 * alpha_1 * y(0) * y(0) * ydot(1);

    return dy;
}

int main(const int argc, const char *const argv[]) {
    solve_beam_ode_explicit();

    plt::show();

    return 0;
}
