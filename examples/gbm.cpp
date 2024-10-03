#pragma optimize("", off)
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

#include <nld/autocont.hpp>

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

int main() {
    using nld::concat;
    using nld::mean_amplitude;
    using nld::monodromy;
    using nld::solution;

    nld::continuation_parameters params(nld::newton_parameters(25, 1.0e-3), 7.9,
                                        0.001, 0.0001, nld::direction::forward);

    auto u0 = nld::vector_xd(2);
    // u0 << 1.162, 1.25;
    // u0 << 1.187979797979798, 1.397;
    auto v0 = nld::vector_xd(2);
    v0 << 0.0, 1.0;

    auto Cm1f = [](const auto lambda) -> nld::matrix_xd {
        return nld::matrix_xd::Zero(2, 2);
    };
    auto C0f = [](const auto lambda) -> nld::matrix_xd {
        nld::matrix_xd C0 = nld::matrix_xd::Zero(2, 2);
        double friction = 0.01;
        C0(0, 0) = friction;
        C0(1, 1) = friction;
        return C0;
    };
    auto Cp1f = [](const auto lambda) -> nld::matrix_xd {
        return nld::matrix_xd::Zero(2, 2);
    };

    auto Km1f = [](const auto lambda) -> nld::matrix_xd {
        nld::matrix_xd Km1 = nld::matrix_xd::Zero(2, 2);
        Km1(0, 0) = 0.4;
        Km1(0, 1) = lambda;
        Km1(1, 0) = lambda;
        Km1(1, 1) = 0.4;
        return 0.5 * Km1;
    };

    auto K0f = [](const auto lambda) -> nld::matrix_xd {
        nld::matrix_xd K0 = nld::matrix_xd::Zero(2, 2);
        K0(0, 0) = 0.5;
        K0(1, 1) = 1.5;
        return K0;
    };

    auto Kp1f = [](const auto lambda) -> nld::matrix_xd {
        nld::matrix_xd Kp1 = nld::matrix_xd::Zero(2, 2);
        Kp1(0, 0) = 0.4;
        Kp1(0, 1) = lambda;
        Kp1(1, 0) = lambda;
        Kp1(1, 1) = 0.4;
        return 0.5 * Kp1;
    };

    std::vector<nld::matrix_maker_fn> Cnf = {Cm1f, C0f, Cp1f};
    std::vector<nld::matrix_maker_fn> Knf = {Km1f, K0f, Kp1f};

    auto m = 2;
    auto gbm = nld::generalized_bolotin_method(Cnf, Knf, m,
                                               nld::resonance_type::harmonic);
    // Subharmonic
    u0 << 1.2, gbm.from_omega_to_omega_hat(1.25);
    // u0 << 6.0803, gbm.from_omega_to_omega_hat(1.5802);

    std::vector<double> lambdas0;
    std::vector<double> omegas0;
    params.direction = nld::direction::reverse;
    for (auto [lambda, omega] :
         arc_length(gbm, params, u0, v0, nld::point2d(0, 1))) {
        lambdas0.push_back(lambda);
        auto omega_real = gbm.from_omega_hat_to_omega(omega);
        omegas0.push_back(omega_real);
        std::cout << "lambda: " << lambda << " ; "
                  << "omega: " << omega_real << std::endl;
    }

    // Subharmonic
    // u0 << 1.30094, gbm.from_omega_to_omega_hat(1.1868);
    // Harmonic
    // u0 << 10.4751, gbm.from_omega_to_omega_hat(0.433805);
    // Combinational
    // u0 << 1.2, gbm.from_omega_to_omega_hat(1.25);
    u0 << 0.3575, gbm.from_omega_to_omega_hat(0.75);
    std::cout << "u0: " << u0 << std::endl;

    std::vector<double> lambdas1;
    std::vector<double> omegas1;
    params.direction = nld::direction::reverse;
    for (auto [lambda, omega] :
         arc_length(gbm, params, u0, v0, nld::point2d(0, 1))) {
        lambdas1.push_back(lambda);
        auto omega_real = gbm.from_omega_hat_to_omega(omega);
        omegas1.push_back(omega_real);
        if (std::abs(lambda - 1.2) < 1e-3)
            std::cout << "lambda12: " << lambda << " ; "
                      << "omega12: " << omega_real << std::endl;
        std::cout << "lambda: " << lambda << " ; "
                  << "omega: " << omega_real << std::endl;
    }

    plt::plot(omegas0, lambdas0);
    plt::plot(omegas1, lambdas1);
    plt::xlim(-3.0, 3.0);
    plt::ylim(-5.0, 5.0);
    plt::show();

    std::cout << "m: " << m << "\n";

    //
    // auto x = s(0);
    // auto y = s(1);
    //
    // plt::plot(x, y);
    // plt::show();

    return 0;
}
