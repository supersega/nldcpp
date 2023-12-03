constexpr auto PI = 3.14159265358979323846264338327950288;
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>
#include <nld/collocations.hpp>
using namespace nld;

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

// Duffing oscilator with energy dissipation
vector_xdd duffing(const vector_xdd &y, dual t, const vector_xdd &parameters) {
    vector_xdd dy(y.size());

    dual t8 = cos(2.0 * PI * t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - 0.8600261454e-2 * t8;

    auto T = parameters(0);
    dy *= T;

    return dy;
}

auto initial_guess(nld::collocations::mesh_parameters parameters,
                   std::size_t dimension) -> nld::vector_xdd {
    nld::vector_xdd u0(dimension);
    u0 << -7.51716e-05, 0.000116238;

    auto N = parameters.intervals;
    auto m = parameters.collocation_points;
    std::size_t n = dimension;

    nld::vector_xdd u(N * n * m + (N - 1) * n + n + 1);
    u.head(n) = u0;
    u(u.size() - 1) = 2.0 * M_PI / 0.2;

    double t = 0.0;
    double dt = 1.0 / (N * (m + 1));
    std::cout << "dt = " << dt << std::endl;
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < m + 1; ++k) {
            auto idx = j * (m + 1) * n + k * n;
            // std::cout << "idx = " << idx << std::endl;

            u.segment(j * (m + 1) * n + k * n, n) = u0;
            u0 = nld::math::runge_kutta_4::end_solution(
                duffing, nld::math::constant_step_parameters{t, t + dt, 1}, u0,
                std::tuple(nld::vector_xdd{u.tail(1)}));
            t += dt;
        }
        if (j < N - 1) {
            u.segment(j * (m + 1) * n + (m)*n, n) = u0;
        }
    }

    std::cout << "t = " << t << std::endl;

    return u;
}

int main() {
    //     continuation_parameters params(newton_parameters(25, 0.00001), 45.1,
    //     0.001,
    //                                    0.01, direction::reverse);
    //
    //     nld::collocations::mesh_parameters mesh_params{30, 3};
    //     auto basis_builder = nld::collocations::make_basis_builder<
    //         nld::collocations::lagrange_basis>();
    //     auto bc = [](const auto &u0, const auto &u1) { return u0 - u1; };
    //
    //     nld::collocations::boundary_value_problem system(duffing, bc,
    //     basis_builder,
    //                                                      mesh_params, 2);
    //
    //     auto u0 = initial_guess(mesh_params, 2);
    //     nld::vector_xdd v0 = nld::vector_xdd::Zero(u0.size());
    //     v0(v0.size() - 1) = 1.0;
    //
    //     std::vector<double> A1;
    //     std::vector<double> Omega;
    //     for (auto v : arc_length(system, params, u0, v0, solution())) {
    //         nld::vector_xd u_0((v.size() - 1) / 2);
    //         for (std::size_t i = 0; i < u_0.size(); ++i) {
    //             u_0[i] = (double)v[i * 2];
    //         }
    //
    //         auto max = u_0.maxCoeff();
    //         auto min = u_0.minCoeff();
    //         auto A1_ = (max - min) / 2.0;
    //
    //         auto ut = v.head(v.size() - 1);
    //         auto omega = 2.0 * PI / (double)v(v.size() - 1);
    //         Omega.push_back((double)omega);
    //         A1.push_back((double)A1_);
    //         std::cout << "A1 = " << v(0) << ", omega = " << omega <<
    //         std::endl;
    //     }
    //
    //     plt::ylabel(R"($A_1$)");
    //     plt::xlabel(R"($\Omega$)");
    //     plt::named_plot("Duffing without explicit frequence/period argument",
    //     Omega,
    //                     A1, "-b");
    //     plt::show();
}
