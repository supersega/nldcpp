
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>
using namespace nld;

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

// Duffing oscilator with energy dissipation
vector_xdd duffing(const vector_xdd &y, dual t) {
    vector_xdd dy(y.size());

    dual t8 = cos(2.0 * PI * t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - 0.8600261454e-2 * t8;

    return dy;
}

int main() {
    continuation_parameters params(newton_parameters(25, 0.00001), 35.1, 0.001,
                                   0.01, direction::reverse);

    auto ip = periodic_parameters_adaptive{1, 1.0 / 200, 1.0 / 20.0, 2.0e-6};
    // auto ip = periodic_parameters_constant{1, 200};
    auto bvp = periodic<runge_kutta_45>(non_autonomous(duffing), ip);

    vector_xdd ys(3);
    ys << 0.0, 0.0, 2.0 * PI / 0.2;

    vector_xdd vs(3);
    vs << 0.0, 0.0, 1.0;

    std::vector<double> A1;
    std::vector<double> Omega;
    for (auto [v, A] : arc_length(bvp, params, ys, vs,
                                  concat(solution(), mean_amplitude(0)))) {
        auto omega = 2.0 * PI / (double)v[2];
        Omega.push_back((double)omega);
        A1.push_back(abs((double)v[0]));
    }

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::named_plot("Duffing without explicit frequence/period argument", Omega,
                    A1, "-b");
    plt::show();
}
