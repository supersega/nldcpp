constexpr auto PI = 3.14159265358979323846264338327950288;
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

// Duffing oscilator with energy dissipation
nld::vector_xdd duffing(const nld::vector_xdd &y, nld::dual t, nld::dual Omega,
                        nld::dual A) {
    nld::vector_xdd dy(y.size());

    nld::dual t8 = cos(t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - A * 0.8600261454e-2 * t8;

    dy *= 2.0 * PI;
    dy /= Omega;

    return dy;
}

int main() {
    using namespace std::placeholders;
    for (std::size_t i = 1; i < 25; i += 5) {
        if (i == 6)
            i = 5;

        nld::continuation_parameters params(
            nld::newton_parameters(25, 0.000005), 10.1, 0.01, 0.01,
            nld::direction::forward);

        auto ip = nld::periodic_parameters_constant{1, 200};
        nld::dual A = 1.0 * i;
        auto bvp = periodic<nld::runge_kutta_4>(
            nld::non_autonomous(std::bind(duffing, _1, _2, _3, A)), ip);

        nld::vector_xdd ys(3);
        ys << 0.0, 0.0, 0.37;

        nld::vector_xdd vs(3);
        vs << 0.0, 0.0, 1.0;

        std::vector<double> A1;
        std::vector<double> Omega;
        std::stringstream ss;
        ss << "/Volumes/Data/phd2023/Free vibrations/AFC_" << i << ".txt";
        ofstream fs(ss.str(), std::ofstream::trunc);
        for (auto [v, A] :
             arc_length(bvp, params, ys, vs,
                        concat(nld::solution(), nld::mean_amplitude(0)))) {
            Omega.push_back((double)v[2]);
            A1.push_back((double)A);

            fs << Omega.back() << ' ' << A1.back() << std::endl;
        }
    }

    nld::continuation_parameters params(nld::newton_parameters(25, 0.000005),
                                        15.1, 0.01, 0.01,
                                        nld::direction::forward);

    auto ip = nld::periodic_parameters_constant{1, 200};
    nld::dual omega = 0.4;
    auto bvp = periodic<nld::runge_kutta_4>(
        nld::non_autonomous(std::bind(duffing, _1, _2, omega, _3)), ip);

    nld::vector_xdd ys(3);
    ys << 0.0, 0.0, 0.33;

    nld::vector_xdd vs(3);
    vs << 0.0, 0.0, 1.0;

    std::vector<double> A1;
    std::vector<double> Force;
    std::stringstream ss;
    ss << "/Volumes/Data/phd2023/Free vibrations/AAC"
       << ".txt";
    ofstream fs(ss.str(), std::ofstream::trunc);
    for (auto [v, A] :
         arc_length(bvp, params, ys, vs,
                    concat(nld::solution(), nld::mean_amplitude(0)))) {
        Force.push_back((double)v[2]);
        A1.push_back((double)A);

        fs << Force.back() << ' ' << A1.back() << std::endl;
    }
}
