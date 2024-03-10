constexpr auto PI = 3.14159265358979323846264338327950288;
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

/// @brief Conservative system with periodic solutions
/// @details This system has periodic solutions, in examples
/// of AUTO it described as "Model with vectical Hopf"
/// and we not involve 'lambda' into continuation process, but
/// just use period
auto conservative(const nld::vector_xdd &u) {
    nld::vector_xdd f(2);

    f[0] = 0 - u[1];
    f[1] = u[0] - u[0] * u[0];

    return f;
}

/// @brief Conservative system with periodic solutions
/// @details This system has periodic solutions, in examples
/// of AUTO it described as "Model with vectical Hopf"
/// and we involve 'lambda' into continuation process, but
/// so poincare equation is needed
auto conservative_with_parameter(const nld::vector_xdd &u, nld::dual lambda) {
    nld::vector_xdd f(2);

    f[0] = lambda - u[1];
    f[1] = u[0] - u[0] * u[0];

    return f;
}

int main() {
    using nld::concat;
    using nld::mean_amplitude;
    using nld::monodromy;
    using nld::solution;

    nld::continuation_parameters params(nld::newton_parameters(25, 0.0001), 4.1,
                                        0.003, 0.00025,
                                        nld::direction::forward);

    auto ip = nld::periodic_parameters_constant{1, 300};
    auto bvp =
        nld::periodic<nld::runge_kutta_4>(nld::autonomous(conservative), ip);

    nld::vector_xdd u0(3);
    u0 << 0.0966822, 0.0, 6.30641;

    nld::vector_xdd v0(3);
    v0 << 0.0, 0.0, 1.0;

    std::vector<double> Am;
    std::vector<double> L;
    std::cout << "Start\n";
    for (auto [solution, monodromy, amplitude] :
         arc_length(bvp, params, u0, v0,
                    concat(solution(), monodromy(), mean_amplitude(0)))) {
        auto period = solution(solution.size() - 1);
        std::cout << solution << std::endl;
        std::cout << "Frq: " << period << " Amplitude: " << amplitude
                  << std::endl;

        Am.push_back(double(amplitude));
        L.push_back(double(period));
    };

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($T$)");
    // plt::xlim(6.25, 8.0);
    plt::named_plot("Saddle Node", L, Am, "-b");
    plt::show();
}
