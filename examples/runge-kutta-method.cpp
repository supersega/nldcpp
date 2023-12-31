// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <iostream>

#include <nld/math.hpp>

nld::vector_xdd ode_p(const nld::vector_xdd &y, nld::dual t, nld::dual Omega) {
    nld::vector_xdd dy(y.size());

    nld::dual t5 = pow(y[0], 0.2e1);
    nld::dual t8 = cos(t);

    dy[0] = y[1] / Omega;
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * t5 * y[0] - 0.8600261454e-3 * t8;

    return dy;
}

auto ode(const nld::vector_xdd &y, nld::dual t) { return ode_p(y, t, 1.3); }

auto ode2(const nld::vector_xdd &y, nld::dual t) {
    nld::vector_xdd dy(y.size());

    dy[0] = 1 + y[0] * y[0];

    return dy;
}

int main() {
    nld::vector_xdd x = nld::vector_xdd::Zero(1);

    auto end4 = nld::runge_kutta_4::end_solution(
        ode2, nld::constant_step_parameters{0.0, 1.4, 14}, x);
    auto end45 = nld::runge_kutta_45::end_solution(
        ode2, nld::adaptive_step_parameters{0.0, 1.4, 0.001, 0.1, 2.0e-5}, x);

    std::cout << "Solution rk4 is : " << end4 << '\n';
    std::cout << "Solution rk45 is : " << end45 << '\n';
}
