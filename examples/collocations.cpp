#include <Eigen/Sparse>
#include <iostream>
#include <matplotlibcpp.h>
#include <nld/autocont.hpp>
#include <nld/collocations.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>
#include <utility>
#include <vector>
#define _USE_MATH_DEFINES
#include <chrono>
#include <math.h>

namespace plt = matplotlibcpp;

auto conservative(const nld::vector_xdd &u, nld::dual t) -> nld::vector_xdd {
    nld::vector_xdd f(2);

    f[0] = 0 - u[1];
    f[1] = u[0] - u[0] * u[0];

    auto T = 2.0 * M_PI;
    f *= T;

    return f;
}

/// @brief First order system
/// @details from:
/// https://math.stackexchange.com/questions/1656115/solve-first-order-differential-equation-boundary-value-problem-using-matlab
/// bc: u(0) = -u(1)
/// TODO: as a first step compare shooting and collocations
auto first_order(const nld::vector_xdd &u, nld::dual t) -> nld::vector_xdd {
    nld::vector_xdd f(1);

    double a = 160.0;
    double b = 6500.0;
    double m = 30.0;
    double k = 700.0;

    auto a1 = (-a * t + b) / (-t * t + t + m);
    auto a2 = k / (-t * t + t + m);

    f[0] = -a1 * u[0] - a2;

    return f;
}

template <typename F>
struct shooting_system_impl {
    explicit shooting_system_impl(F &&f) : function(std::forward<F>(f)) {}

    auto operator()(const nld::vector_xdd &u0) const -> nld::vector_xdd {

        nld::vector_xdd u0_ = u0.head(u0.size() - 1);
        auto u1 = nld::math::runge_kutta_4::end_solution(
            function, nld::math::constant_step_parameters{0.0, 1.0, 300}, u0_,
            std::tuple(nld::vector_xdd{u0.tail(1)}));

        nld::vector_xdd f = u0_ - u1;

        return f;
    }

    F function;
};

template <typename F>
shooting_system_impl(F &&f) -> shooting_system_impl<F>;

auto initial_guess(nld::collocations::mesh_parameters parameters,
                   std::size_t dimension) -> nld::vector_xdd {
    nld::vector_xdd u0(dimension);
    u0 << -7.51716e-05, 0.000116238;

    auto N = parameters.intervals;
    auto m = parameters.collocation_points;
    std::size_t n = dimension;

    nld::vector_xdd u(N * n * m + (N - 1) * n + n + 1);
    u.head(n) = u0;
    u(u.size() - 1) = 2.0 * M_PI / 0.95;

    double t = 0.0;
    double dt = 1.0 / (N * (m + 1));
    std::cout << "dt = " << dt << std::endl;
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < m + 1; ++k) {
            auto idx = j * (m + 1) * n + k * n;
            // std::cout << "idx = " << idx << std::endl;

            u.segment(j * (m + 1) * n + k * n, n) = u0;
            u0 = nld::math::runge_kutta_4::end_solution(
                conservative, nld::math::constant_step_parameters{t, t + dt, 1},
                u0);
            t += dt;
        }
        if (j < N - 1) {
            u.segment(j * (m + 1) * n + (m)*n, n) = u0;
        }
    }

    std::cout << "t = " << t << std::endl;

    return u;
}

nld::vector_xdd duffing(const nld::vector_xdd &y, nld::dual t,
                        const nld::vector_xdd &p) {
    nld::vector_xdd dy(y.size());

    nld::dual t8 = cos(2.0 * M_PI * t);

    dy[0] = y[1];
    dy[1] = -0.1e-1 * y[1] - 0.1000000000e1 * y[0] -
            0.1499999998e2 * y[0] * y[0] * y[0] - 0.8600261454e-2 * t8;

    auto T = p(0);
    dy *= T;

    return dy;
}

auto interpolate(auto basis_builder, const nld::collocations::mesh &mesh,
                 const nld::vector_xd &u, std::size_t K = 21)
    -> std::vector<double> {
    auto N = mesh.nodes.size() - 1;
    auto M = mesh.collocation_points.size();
    std::size_t m = M / N;

    std::vector<double> f;
    for (std::size_t i = 0; i < N; ++i) {
        nld::segment element{mesh.nodes[i], mesh.nodes[i + 1]};
        auto basis = basis_builder(element, m);
        nld::vector_xd u_i = u.segment(i * (m + 1), m + 1);
        auto polynomial = basis.interpolate(u_i);

        for (std::size_t j = 0; j < K - 1; ++j) {
            auto t =
                element.begin + j * (element.end - element.begin) / (K - 1);
            f.push_back(polynomial(t));
        }

        if (i == N - 1) {
            auto t = element.end;
            f.push_back(polynomial(t));
        }
    }

    return f;
}

int main(int argc, char *argv[]) {
    std::cout << "Collocations!" << std::endl;

    nld::collocations::mesh_parameters parameters{90, 3};
    nld::collocations::mesh mesh(
        parameters, nld::collocations::uniform_mesh_nodes,
        nld::collocations::legandre_collocation_points);

    auto basis_builder = nld::collocations::make_basis_builder<
        nld::collocations::lagrange_basis>();
    nld::collocations::collocation_on_elements collocation_on_elements(
        mesh, basis_builder);

    auto np = nld::math::newton_parameters(10, 0.00005);
    nld::vector_xdd u(3);
    u[0] = -7.51716e-05;
    u[1] = 0.000116238;
    u[2] = 2.0 * M_PI / 0.95;
    auto sh = shooting_system_impl(duffing);
    if (auto info = nld::math::newton(sh, wrt(u.head(u.size() - 1)), at(u), np);
        info) {
        std::cout << "Iterations done = " << info.number_of_done_iterations
                  << '\n';
        std::cout << "Great work u = " << u << '\n';
    }

    auto bc = [](const auto &u0, const auto &u1) { return u0 - u1; };

    nld::collocations::boundary_value_problem system(duffing, bc, basis_builder,
                                                     parameters, 2);
    auto u0 = initial_guess(parameters, 2);
    auto f = system(u0);

    auto now = std::chrono::high_resolution_clock::now();

    nld::vector_xdd V;
    // auto J = system.jacobian(u0, V);

    // std::cout << "Jacobian = \n" << nld::matrix_xd(J) << std::endl;

    // auto J_autodiff = system.jacobian(nld::wrt(u0), nld::at(u0), V);

    // std::cout << "Jacobian autodiff = \n" << J_autodiff << std::endl;

    auto wrt = u0.head(u0.size() - 1);
    if (auto info = nld::math::newton(system, nld::wrt(wrt), nld::at(u0), np);
        info) {
        std::cout << "Iterations done = " << info.number_of_done_iterations
                  << '\n';
        std::cout << "Great work: colobok u = " << u0.head(2) << '\n';
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       now)
                     .count()
              << " ms\n";

    {
        nld::vector_xdd u_ = u.head(u.size() - 1);
        auto solution = nld::math::runge_kutta_4::solution(
            duffing, nld::math::constant_step_parameters{0.0, 1.0, 300}, u_,
            std::tuple(nld::vector_xdd{u.tail(1)}));

        nld::vector_xd xE = solution.col(0);
        nld::vector_xd yE = solution.col(1);
        std::vector<double> y(yE.data(), yE.data() + yE.size());
        std::vector<double> x(xE.data(), xE.data() + xE.size());
        std::vector<double> t(y.size());
        for (std::size_t i = 0; i < y.size(); ++i) {
            t[i] = i * 1.0 / (y.size() - 1);
        }

        plt::named_plot("shooting curve", x, y);
    }
    {
        nld::vector_xd u_0((u0.size() - 1) / 2);
        nld::vector_xd u_1((u0.size() - 1) / 2);
        for (std::size_t i = 0; i < u_0.size(); ++i) {
            u_0[i] = (double)u0[i * 2];
            u_1[i] = (double)u0[i * 2 + 1];
        }

        auto x_colobok = interpolate(basis_builder, mesh, u_0, 10);
        auto y_colobok = interpolate(basis_builder, mesh, u_1, 10);

        plt::named_plot("collocation curve", x_colobok, y_colobok, "--");
    }

    plt::title("Duffing compare");
    plt::legend();
    plt::show();

    return 0;
}
