// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

nld::vector_xdd2 NLTVA(const nld::vector_xdd2 &y, nld::dual2 t,
                       const nld::vector_xdd2 &params) {
    nld::vector_xdd2 dy(4);

    auto epsilon = 0.05;
    auto m1 = 1.0;
    auto c1 = 0.002;
    auto k1 = 1.0;
    auto knl1 = 1.0;
    auto k2u = 8.0 * epsilon *
               (16.0 + 23.0 * epsilon + 9.0 * epsilon * epsilon +
                (4.0 + 2.0 * epsilon) * sqrt(4.0 + 3.0 * epsilon));
    auto k2l = 3.0 * (1.0 + epsilon) * (1.0 + epsilon) *
               (64.0 + 80.0 * epsilon + 27.0 * epsilon * epsilon);
    auto k2 = k2u / k2l;
    auto knl2 = 2.0 * epsilon * epsilon * knl1 / (1.0 + 4 * epsilon);
    auto caux = (k2 * m1 * epsilon *
                 (8.0 + 9.0 * epsilon - 4.0 * sqrt(4.0 + 3.0 * epsilon))) /
                4.0 / (1.0 + epsilon);
    auto c2 = sqrt(caux);

    auto f0 = params(1);

    dy[0] = y[2];
    dy[1] = y[3];
    dy[2] = -c1 * y[2] - k1 * y[0] - knl1 * y[0] * y[0] * y[0] -
            c2 * (y[2] - y[3]) - k2 * (y[0] - y[1]) -
            knl2 * (y[0] - y[1]) * (y[0] - y[1]) * (y[0] - y[1]) +
            f0 * cos(t * 2.0 * PI);
    dy[3] = -c2 * (y[3] - y[2]) - k2 * (y[1] - y[0]) -
            knl2 * (y[1] - y[0]) * (y[1] - y[0]) * (y[1] - y[0]);

    dy[3] = dy[3] / epsilon;

    dy *= 2.0 * PI;
    dy /= params(0);

    return dy;
}

int main() {
    nld::continuation_parameters params(nld::newton_parameters(25, 0.0001),
                                        15.5, 0.01, 0.01,
                                        nld::direction::reverse);

    auto ip = nld::periodic_parameters_constant{1, 200};
    auto snb = saddle_node<nld::runge_kutta_4>(nld::non_autonomous(NLTVA), ip);

    nld::vector_xdd2 u0(10);
    u0 << 0.0372658, -2.25952, 0.781638, 2.70618, 0.158379, 0.601359, 0.0393662,
        0.782134, 1.11126, 0.15;

    nld::vector_xdd2 v0(10);
    v0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

    ofstream fs("saddle_node.txt", std::ofstream::trunc);

    std::vector<double> Ai;
    std::vector<double> A0i;
    for (auto [s, A0] :
         arc_length(snb, params, u0, v0,
                    concat(nld::solution(), nld::mean_amplitude(0)))) {
        auto A = s(9);
        auto Omega = s(8);
        std::cout << "Solution = \n" << A << endl;
        fs << A << ' ' << A0 << ' ' << Omega << endl;
        Ai.push_back(static_cast<double>(Omega));
        A0i.push_back(static_cast<double>(A0));
    }

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($f_0$)");
    plt::named_plot("Saddle Node", Ai, A0i, "-b");
    plt::show();
}
