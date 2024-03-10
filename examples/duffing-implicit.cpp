#include <fstream>
#include <iostream>
#include <vector>
#include <numbers>
#include <nld/autocont.hpp>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

// Duffing oscilator with energy dissipation parameter is not defined
// explicitly, so period is assumed as continuation parameter.
nld::vector_xdd duffing(const nld::vector_xdd &y, nld::dual t) {
    nld::vector_xdd dy(y.size());

    nld::dual t8 = cos(2.0 * std::numbers::pi * t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - 0.8600261454e-2 * t8;

    return dy;
}

int main() {
    // Create continuation parameters with next parameters:
    // 1. Newton parameters with 25 iterations and 0.00001 tolerance
    // 2. Maximum arclength of 35.1
    // 3. Minimal arclength step of 0.001
    // 4. Maximal arclength step of 0.01
    // 5. Direction of continuation is reverse (according to continuation parameter)
    nld::continuation_parameters params(nld::newton_parameters(25, 0.00001),
                                        35.1, 0.001, 0.01,
                                        nld::direction::reverse);

    // Create periodic parameters for solver with adaptive stepsize:
    // 1. One period
    // 2. 300 steps
    auto ip =
        nld::periodic_parameters_constant{1, 300};

    // Create boundary value problem with simple shooting method
    // and runge_kutta_4 as integrator with next parameters:
    // 1. Duffing oscillator as non-autonomous system. It is necessary to
    //   define type of system, because autonomous or non-autonomous might
    //   have same signature.
    // 2. Periodic parameters created before
    auto bvp = periodic<nld::runge_kutta_4>(nld::non_autonomous(duffing), ip);

    // Initial guess for periodic solution in next form:
    // 1. Initial condition for y, wnere y in R^2
    // 2. Initial condition for period
    nld::vector_xdd ys(3);
    ys << 0.0, 0.0, 2.0 * std::numbers::pi / 0.2;

    // Initial tangent in next form:
    // 1. Zero tangent for y
    // 2. One for period
    nld::vector_xdd vs(3);
    vs << 0.0, 0.0, 1.0;

    // Create vectors for storing amplitude and frequency
    std::vector<double> A1;
    std::vector<double> Omega;

    // Run continuation with next parameters:
    // 1. Boundary value problem
    // 2. Continuation parameters
    // 3. Initial conditions
    // 4. Initial tangent
    // 6. Mapper functions which tells what to store on iteration.
    //   Explanation of mapper functions:
    //   - nld::solution() -> T1 - function to store solution
    //   - nld::mean_amplitude(0) -> T2 - function to store mean amplitude of solution
    //   - cooncat - function to concatenate several mappers result, e.g. return std::tuple<T1, T2>
    for (auto [v, A] :
         arc_length(bvp, params, ys, vs,
                    concat(nld::solution(), nld::mean_amplitude(0)))) {
        auto omega = 2.0 * std::numbers::pi / (double)v[2];
        Omega.push_back(omega);
        A1.push_back(A);
    }

    // Plot amplitude and frequency
    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::named_plot("Duffing without explicit frequence/period argument", Omega,
                    A1, "-b");
    plt::show();
}
