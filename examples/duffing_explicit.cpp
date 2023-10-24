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
vector_xdd duffing(const vector_xdd &y, dual t, dual Omega) {
    vector_xdd dy(y.size());

    dual t8 = cos(t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - 0.8600261454e-2 * t8;

    dy *= 2.0 * PI;
    dy /= Omega;

    return dy;
}

int main() {
    continuation_parameters params(newton_parameters(25, 0.000005), 6.1, 0.003,
                                   0.003, direction::forward);

    auto ip = periodic_parameters{1, 200};
    auto bvp = periodic<runge_kutta_4>(non_autonomous(duffing), ip);

    vector_xdd ys(3);
    ys << 0.0, 0.0, 0.4;

    vector_xdd vs(3);
    vs << 0.0, 0.0, 1.0;

    std::vector<double> A1;
    std::vector<double> Omega;
    for (auto [v, A] : arc_length(bvp, params, ys, vs,
                                  concat(solution(), mean_amplitude(0)))) {
        Omega.push_back((double)v[2]);
        A1.push_back((double)A);
    }

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::named_plot("Duffing with explicit frequence/period argument", Omega,
                    A1, "-b");
    plt::show();
}
