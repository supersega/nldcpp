// C++ includes
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

    dual t5 = pow(y[0], 0.2e1);
    dual t8 = cos(t);

    dy[0] = y[1] / Omega;
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * t5 * y[0] - 0.8600261454e-3 * t8;

    return dy;
}

vector_xdd ABC_reaction(const vector_xdd &u, dual T, dual D) {
    vector_xdd du(u.size());

    double alpha = 1;
    double sigma = 0.04;
    double B = 8;
    double betta = 1.55;

    dual du_0 = -u(0) + D * (1 - u(0)) * exp(u(2));
    dual du_1 =
        -u(1) + D * (1 - u(0)) * exp(u(2)) + D * sigma * u(1) * exp(u(2));
    dual du_2 = -u(2) * betta * u(2) + D * B * (1 - u(0)) * exp(u(2)) +
                D * B * sigma * alpha * u(1) * exp(u(2));

    du(0) = du_0;
    du(1) = du_1;
    du(2) = du_2;

    du *= T;

    return du;
}

vector_xdd ABC_reaction_0(const vector_xdd &u, dual D) {
    vector_xdd du(u.size());

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

vector_xdd f(const vector_xdd &u, dual T, dual alpha) {
    vector_xdd f(u.size());

    f[0] = -2.0 * u[0] + u[1] + alpha * exp(u[0]);
    f[1] = u[0] - 2.0 * u[1] + alpha * exp(u[1]);

    return f;
}

auto simple_system(const vector_xdd &u, dual T, dual lambda) {
    vector_xdd f(u.size());

    f[0] = (1.0 - lambda) * u[0] - u[1];
    f[1] = u[0] + u[0] * u[0];

    f *= T;

    return f;
}

auto simple_system_2(const vector_xdd &u, dual lambda) {
    vector_xdd f(2);

    f[0] = (1.0 - lambda) * u[0] - u[1];
    f[1] = u[0] + u[0] * u[0];

    return f;
}

auto simple_system_3(const vector_xdd &u, dual lambda) {
    vector_xdd f(2);

    f[0] = (lambda)*u[0] - u[1];
    f[1] = u[0] - u[0] * u[0];

    return f;
}

auto duffing_autonomous(const vector_xdd &y, dual omega) {
    vector_xdd dy(4);

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
    ofstream fs("/Volumes/Data/dev/nonlinear-dynamic-phd/src/3rd-year-report/"
                "data/stuff.csv");
    fs << 'x' << ';' << 'y' << endl;

    continuation_parameters params(newton_parameters(25, 0.000005), 2.1, 0.003,
                                   0.003, direction::forward);

    auto ip = periodic_parameters{1, 200};
    auto bvp = periodic<runge_kutta_4>(autonomous(simple_system_3), ip);

    vector_xdd u0(2);
    u0 << 0.0, 0.0;
    dual lambda0 = 0.0;
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

    vector_xdd u(2);
    dual T = 2.0 * PI;
    dual lambda = 0.0;
    auto s = 0.002;

    u << 0.0, 0.0;

    vector_xdd us(4);
    us << u, T, lambda;

    vector_xdd vs(4);
    vs << -nc, 0.0, 0.0;

    std::vector<double> Am;
    std::vector<double> L;
    vector_xdd v0;
    dual lmbd = 0;
    dual per = 0;
    for (auto [v, A] : arc_length(bvp, params, us, vs,
                                  concat(solution(), mean_amplitude(0)))) {
        v0 = v.head(2);
        lmbd = v[3];
        per = v[2];
        L.push_back((double)per);
        Am.push_back((double)A);
    }

    std::cout << "T: " << per << std::endl;
    std::cout << "v0: " << v0 << std::endl;

    auto sol = runge_kutta_4::solution(
        [lmbd](const auto &y, auto t) { return simple_system_3(y, lmbd); },
        constant_step_parameters{0.0, 3.0 * (double)per, 2000}, v0);

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
