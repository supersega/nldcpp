// C++ includes
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <fstream>
#include <iostream>
#include <nld/autocont.hpp>
#include <vector>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;
// Duffing oscilator with energy dissipation
nld::vector_xdd duffing(const nld::vector_xdd &y, nld::dual t,
                        nld::dual Omega) {
    nld::vector_xdd dy(y.size());

    nld::dual t5 = pow(y[0], 0.2e1);
    nld::dual t8 = cos(t);

    dy[0] = y[1] / Omega;
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * t5 * y[0] - 0.8600261454e-3 * t8;

    return dy;
}

nld::vector_xdd ABC_reaction(const nld::vector_xdd &u, nld::dual T,
                             nld::dual D) {
    nld::vector_xdd du(u.size());

    double alpha = 1;
    double sigma = 0.04;
    double B = 8;
    double betta = 1.55;

    nld::dual du_0 = -u(0) + D * (1 - u(0)) * exp(u(2));
    nld::dual du_1 =
        -u(1) + D * (1 - u(0)) * exp(u(2)) + D * sigma * u(1) * exp(u(2));
    nld::dual du_2 = -u(2) * betta * u(2) + D * B * (1 - u(0)) * exp(u(2)) +
                     D * B * sigma * alpha * u(1) * exp(u(2));

    du(0) = du_0;
    du(1) = du_1;
    du(2) = du_2;

    du *= T;

    return du;
}

nld::vector_xdd ABC_reaction_0(const nld::vector_xdd &u, nld::dual D) {
    nld::vector_xdd du(u.size());

    double alpha = 1;
    double sigma = 0.04;
    double B = 8;
    double betta = 1.20;

    du(0) = -u(0) + D * (1 - u(0)) * exp(u(2));
    du(1) = -u(1) + D * (1 - u(0)) * exp(u(2)) + D * sigma * u(1) * exp(u(2));
    du(2) = -u(2) * betta * u(2) + D * B * (1 - u(0)) * exp(u(2)) +
            D * B * sigma * alpha * u(1) * exp(u(2));

    return du;
}

nld::vector_xdd f(const nld::vector_xdd &u, nld::dual T, nld::dual alpha) {
    nld::vector_xdd f(u.size());

    f[0] = -2.0 * u[0] + u[1] + alpha * exp(u[0]);
    f[1] = u[0] - 2.0 * u[1] + alpha * exp(u[1]);

    return f;
}

auto simple_system(const nld::vector_xdd &u, nld::dual T, nld::dual lambda) {
    nld::vector_xdd f(u.size());

    f[0] = (1.0 - lambda) * u[0] - u[1];
    f[1] = u[0] + u[0] * u[0];

    f *= T;

    return f;
}

auto simple_system_2(const nld::vector_xdd &u, nld::dual lambda) {
    nld::vector_xdd f(2);

    f[0] = (1.0 - lambda) * u[0] - u[1];
    f[1] = u[0] + u[0] * u[0];

    return f;
}

auto simple_system_3(const nld::vector_xdd &u, nld::dual lambda) {
    nld::vector_xdd f(2);

    f[0] = (lambda)*u[0] - u[1];
    f[1] = u[0] - u[0] * u[0];

    return f;
}

auto duffing_autonomous(const nld::vector_xdd &y, nld::dual omega) {
    nld::vector_xdd dy(4);

    double c = 0.01;
    double k = 1.0;
    double alpha = 0.7;
    double A = 3.0;

    dy(0) = y(1);
    dy(1) = -k * y(0) - c * y(1) - alpha * y(0) * y(0) * y(0) + A * y(3);
    dy(2) = y(2) + omega * y(3) - y(2) * (y(2) * y(2) + y(3) * y(3));
    dy(3) = -omega * y(2) + y(3) - y(3) * (y(2) * y(2) + y(3) * y(3));

    return dy;
}

int main() {
    nld::continuation_parameters params(nld::newton_parameters(25, 0.000005),
                                        2.1, 0.003, 0.003,
                                        nld::direction::forward);

    auto ip = nld::periodic_parameters_constant{1, 200};
    auto bvp =
        periodic<nld::runge_kutta_4>(nld::autonomous(simple_system_3), ip);

    nld::vector_xdd u0(2);
    u0 << 0.0, 0.0;
    nld::dual lambda0 = 0.0;
    auto jacobian_hopf =
        autodiff::forward::jacobian(simple_system_3, wrt(u0), at(u0, lambda0));

    Eigen::EigenSolver<Eigen::MatrixXd> es(jacobian_hopf);
    std::cout << "The eigenvalues of A are:" << std::endl
              << es.eigenvalues() << std::endl;
    std::cout << "The matrix of eigenvectors, V, is:" << std::endl
              << es.eigenvectors() << std::endl
              << std::endl;

    Eigen::VectorXcd v = es.eigenvectors().col(0);
    Eigen::VectorXd nc = v.real();
    Eigen::VectorXd ns = v.imag();

    nld::vector_xdd u(2);
    nld::dual T = 2.0 * PI;
    nld::dual lambda = 0.0;
    auto s = 0.002;

    u << 0.0, 0.0;

    nld::vector_xdd us(4);
    us << u, T, lambda;

    nld::vector_xdd vs(4);
    vs << -nc, 0.0, 0.0;

    std::vector<double> Am;
    std::vector<double> L;
    nld::vector_xdd v0;
    nld::dual lmbd = 0;
    nld::dual per = 0;
    for (auto [v, A] :
         arc_length(bvp, params, us, vs,
                    concat(nld::solution(), nld::mean_amplitude(0)))) {
        v0 = v.head(2);
        lmbd = v[3];
        per = v[2];
        L.push_back((double)per);
        Am.push_back((double)A);
    }

    std::cout << "T: " << per << std::endl;
    std::cout << "v0: " << v0 << std::endl;

    auto sol = nld::runge_kutta_4::solution(
        [lmbd](const auto &y, auto t) { return simple_system_3(y, lmbd); },
        nld::constant_step_parameters{0.0, 3.0 * (double)per, 2000}, v0);

    std::vector<double> xt(sol.rows());
    std::vector<double> ts(sol.rows());
    double dt = 3.0 * (double)per / 2000;
    for (size_t i = 0; i < sol.rows(); i++) {
        xt[i] = sol(i, 0);
        ts[i] = sol(i, 1);
    }

    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($f_0$)");
    plt::xlim(6.25, 8.0);
    plt::named_plot("Saddle Node", L, Am, "-b");
    plt::show();
}
