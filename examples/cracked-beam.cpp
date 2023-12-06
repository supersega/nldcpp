// C++ includes
#include <nld/autocont/mappers.hpp>
constexpr auto PI = 3.14159265358979323846264338327950288;
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include <matplotlibcpp.h>

#include <nld/autocont.hpp>
#include <nld/calculus.hpp>

namespace plt = matplotlibcpp;

struct box {
    auto area() const -> double { return height * width; }

    auto inertia() const -> double {
        auto d = height / 2.0;
        auto b = width / 2.0;
        auto I = 4.0 / 3.0 * b * d * d * d;
        return I;
    }

    double length;
    double height;
    double width;
};

struct physics {
    double young_modulus;
    double density;
};
struct damage {
    double position;
    double depth;
};
struct force {
    double position;
    double amplitude;
};

struct beam final {
    box geometry;
    physics material;
};

// struct caddemi_flexible_beam_dynamic_system final {
//     struct eigen_functions_cracked : nld::basis {
//         eigen_functions_cracked(nld::basis_definition bd, beam b) :
//         nld::basis(bd) {
//
//         }
//
//         auto value(nld::index i) const {
//             auto lambda = beam_.material.young_modulus *
//             beam_.geometry.inertia(); auto [a, b] = interval(i); return [a =
//             a, b = b](auto x) -> adnum { return nld::internal::spline3(x, a,
//             b); };
//         }
//
//         auto subdomain(nld::index i) const -> nld::segment {
//             auto [a, b] = interval(i);
//             auto subdomain = nld::segment { a, b };
//             return *subdomain.intersect(basis::domain());
//         }
//
//         beam beam_;
//     };
// };

struct hinged_beam_bc : nld::boundary_condition {
    explicit hinged_beam_bc(double l) : length(l) {}

    auto value() const {
        return [l = length](auto x) -> nld::adnum { return x * (x - l); };
    }

private:
    double length;
};

struct clampted_hinged_beam_bc : nld::boundary_condition {
    explicit clampted_hinged_beam_bc(double l) : length(l) {}

    auto value() const {
        return [l = length](auto x) -> nld::adnum { return x * x * (x - l); };
    }

private:
    double length;
};

template <typename BC>
struct cracked_flexible_beam_dynamic_system final {
    cracked_flexible_beam_dynamic_system(box g, physics m, damage d, force f,
                                         double fr, std::size_t i, nld::index n,
                                         BC bnd)
        : geometry(g), material(m), crack(d), dofs(n), load(f), friction(fr),
          bc(bnd) {
        calculate_all_terms();
    }

    void calculate_all_terms() {
        std::array dimensions = {Eigen::IndexPair(0, 1)};

        nld::tensor_expression_calculator<nld::gauss_kronrod21> calculator(
            nld::integration_options{1.0e-4, 1.0e-4, 300});

        auto rho = material.density;
        auto A = geometry.height * geometry.width;
        auto E = material.young_modulus;
        auto d = geometry.height / 2.0;
        auto b = geometry.width / 2.0;
        auto h = geometry.height / 2.0;
        auto I = 4.0 / 3.0 * b * d * d * d;
        auto l = geometry.length;
        auto xc = crack.position;
        auto a = crack.depth;
        auto F = load.amplitude;
        auto g = (a / d);
        auto m = 1 / (1.0 + 0.75 * g * g - 1.5 * g - 1.0 / 8.0 * g * g * g);
        auto alpha = 1.276;

        auto domain = nld::segment{0.0, l};
        auto bd = nld::basis_definition{40, {0.0, l}};
        auto bsb = nld::bspline_3_basis(bd);
        auto sp = nld::space<1>{};
        auto &x = sp.coords()[0];
        auto tf = nld::test_functions(sp, bsb);
        auto u = bc * tf;

        auto kinetic_energy_no_crack = nld::integral(rho * A * u * u, domain);
        auto potential_energy_no_crack = nld::integral(
            E * I * nld::laplacian(u) * nld::laplacian(u), domain);
        auto epnc = nld::solve_eigenvalue_problem<nld::gauss_kronrod21>(
            kinetic_energy_no_crack, potential_energy_no_crack,
            nld::integration_options{1.0e-4, 1.0e-4, 300});
        auto efnc = epnc.eigenvalues.cwiseSqrt();
        decltype(auto) eigenvectors_no_crack =
            sqrt(rho * A) * epnc.eigenvectors;

        auto Unc =
            nld::eigenfunctions(u, std::move(eigenvectors_no_crack), dofs);

        auto Wnc =
            nld::integral(Unc * nld::delta_function(load.position), domain);
        auto [Fnc0] = calculator.calculate(Wnc);
        auto Tnc = nld::integral(rho * A * Unc * Unc, domain);
        auto [Mnc] = calculator.calculate(Tnc);
        Eigen::Map<nld::matrix_xd> massnc(Mnc.data(), dofs, dofs);

        nld::matrix_xd Mncinv_matrix = massnc.inverse();
        auto nl_term_no_crack = nld::integral(
            nld::integral(diff(Unc, autodiff::wrt(x)) *
                              diff(Unc, autodiff::wrt(x)),
                          domain) *
                diff(Unc, autodiff::wrt(x)) * diff(Unc, autodiff::wrt(x)),
            domain);
        auto [nl_tensor_no_crack] = calculator.calculate(nl_term_no_crack);

        auto Omega0nc = efnc.minCoeff();

        Omeganc = efnc / Omega0nc;
        auto Gnc0 = (E * A / 2.0 / l) * h * h / Omega0nc / Omega0nc *
                    nl_tensor_no_crack;

        Gnc = Gnc0.contract(nld::utils::tensor_view(Mncinv_matrix), dimensions);
        nld::tensor<1> Fnct =
            F *
            Fnc0.contract(nld::utils::tensor_view(Mncinv_matrix), dimensions) /
            Omega0nc / Omega0nc / h;
        Fnc = Eigen::Map<nld::vector_xd>(Fnct.data(), dofs);

        auto L = [=](auto x) -> nld::adnum {
            return (m - 1.0) * exp(-2.0 * alpha * abs(x - xc) / d);
        };
        auto Q = nld::scalar_function(
            [=](nld::adnum x) -> nld::adnum { return 1.0 / (1.0 + L(x)); });
        auto kinetic_energy_crack = nld::integral(rho * A * u * u, domain);
        auto potential_energy_crack =
            nld::integral(E * I * Q * laplacian(u) * laplacian(u), domain);
        auto epc = nld::solve_eigenvalue_problem<nld::gauss_kronrod21>(
            kinetic_energy_crack, potential_energy_crack,
            nld::integration_options{1.0e-3, 1.0e-3, 500});
        auto efc = epc.eigenvalues.cwiseSqrt();
        decltype(auto) eigenvectors_crack = sqrt(rho * A) * epc.eigenvectors;

        auto Uc = nld::eigenfunctions(u, std::move(eigenvectors_crack), dofs);

        auto Wc =
            nld::integral(Uc * nld::delta_function(load.position), domain);
        auto [Fc0] = calculator.calculate(Wc);
        auto Tc = nld::integral(rho * A * Uc * Uc, domain);
        auto [Mc] = calculator.calculate(Tc);
        Eigen::Map<nld::matrix_xd> massc(Mc.data(), dofs, dofs);
        nld::matrix_xd Mcinv_matrix = massc.inverse();
        auto nl_term_crack = nld::integral(
            nld::integral(diff(Uc, autodiff::wrt(x)) *
                              diff(Uc, autodiff::wrt(x)),
                          domain) *
                diff(Uc, autodiff::wrt(x)) * diff(Uc, autodiff::wrt(x)),
            domain);
        auto [nl_tensor_crack] = calculator.calculate(nl_term_crack);

        Omegac = efc / Omega0nc;
        auto Gc0 =
            (E * A / 2.0 / l) * h * h / Omega0nc / Omega0nc * nl_tensor_crack;
        Gc = Gc0.contract(nld::utils::tensor_view(Mcinv_matrix), dimensions);
        nld::tensor<1> Fct =
            F *
            Fc0.contract(nld::utils::tensor_view(Mcinv_matrix), dimensions) /
            Omega0nc / Omega0nc / h;
        Fc = Eigen::Map<nld::vector_xd>(Fct.data(), dofs);

        auto displacement_in_crack = nld::integral(
            laplacian(Unc) * nld::delta_function(load.position), domain);
        auto [Uxct] = calculator.calculate(displacement_in_crack);
        nld::vector_xd Uxc = Eigen::Map<nld::vector_xd>(Uxct.data(), dofs);

        switch_function = [Uxc, dofs = dofs](const nld::vector_xdd &y) {
            nld::vector_xdd yc = y.head(dofs).template cast<double>();
            return Uxc.cwiseProduct(yc).sum() > 0.0;
        };
    }

    auto closed_crack(const nld::vector_xdd &y, nld::dual t,
                      nld::dual omega) const {
        nld::vector_xdd dy(2 * dofs);
        nld::vector_xdd yc = y.head(dofs);
        auto yct = nld::utils::tensor_view(yc);

        auto f = Omeganc.cwiseProduct(Omeganc).cwiseProduct(yc);

        Eigen::Tensor<nld::dual, 4> G = Gnc.template cast<nld::dual>();
        Eigen::Tensor<nld::dual, 1> fnlt =
            G.contract(yct, std::array{Eigen::IndexPair(3, 0)})
                .contract(yct, std::array{Eigen::IndexPair(2, 0)})
                .contract(yct, std::array{Eigen::IndexPair(1, 0)});
        Eigen::Map<nld::vector_xdd> fnl(fnlt.data(), dofs);

        dy.head(dofs) = y.tail(dofs);
        dy.tail(dofs) = -friction * y.tail(dofs) - f - fnl -
                        Fnc * nld::dual(cos(t * 2.0 * PI));

        dy *= 2.0 * PI;
        dy /= omega;

        return dy;
    }

    auto opened_crack(const nld::vector_xdd &y, nld::dual t,
                      nld::dual omega) const {
        nld::vector_xdd dy(2 * dofs);
        nld::vector_xdd yc = y.head(dofs);
        auto yct = nld::utils::tensor_view(yc);

        auto f = Omegac.cwiseProduct(Omegac).cwiseProduct(yc);

        Eigen::Tensor<nld::dual, 4> G = Gc.template cast<nld::dual>();
        Eigen::Tensor<nld::dual, 1> fnlt =
            G.contract(yct, std::array{Eigen::IndexPair(3, 0)})
                .contract(yct, std::array{Eigen::IndexPair(2, 0)})
                .contract(yct, std::array{Eigen::IndexPair(1, 0)});
        Eigen::Map<nld::vector_xdd> fnl(fnlt.data(), dofs);

        dy.head(dofs) = y.tail(dofs);
        dy.tail(dofs) = -friction * y.tail(dofs) - f - fnl -
                        Fc * nld::dual(cos(t * 2.0 * PI));

        dy *= 2.0 * PI;
        dy /= omega;

        return dy;
    }

    auto operator()(const nld::vector_xdd &y, nld::dual t,
                    nld::dual omega) const {
        if (switch_function(y))
            return opened_crack(y, t, omega);
        else
            return closed_crack(y, t, omega);
    }

    using switch_fn = std::function<bool(const nld::vector_xdd &)>;

    box geometry;     ///< Beam geometry.
    physics material; ///< Beam material.
    damage crack;     ///< Crack position and depth.
    nld::index dofs;  ///< Degrees of freedom number.
    force load;       ///< Load applied to beam.
    double friction;  ///< Friction coefficient.
    BC bc;            ///< Boundary conditions.

    nld::vector_xd
        Omeganc; ///< Dimensionless frequencies for Euler-Bernoulli beam.
    Eigen::Tensor<double, 4>
        Gnc; ///< Nonlinear dimensionless term for Euler-Bernoulli beam.
    nld::vector_xd Fnc; ///< Force amplitude for generalized coordinates in load
                        ///< point for Euler-Bernoulli beam.

    nld::vector_xd Omegac; ///< Dimensionless frequencies for Shen beam.
    Eigen::Tensor<double, 4>
        Gc;            ///< Nonlinear dimensionless term for Shen beam.
    nld::vector_xd Fc; ///< Force amplitude for generalized coordinates in load
                       ///< point for Shen beam.

    switch_fn switch_function; ///< Switch function
};

auto afc(std::size_t i) {
    using nld::concat;
    using nld::mean_amplitude;
    using nld::monodromy;
    using nld::solution;

    box geometry{0.177, 0.01, 0.01};
    physics material{2.1e11, 7800};
    force load{geometry.length / 2.0, 4'000.0};
    double friction = 0.1;
    double coeff = 0.4 * i;
    hinged_beam_bc bc(geometry.length);
    clampted_hinged_beam_bc bc2(geometry.length);

    {
        cracked_flexible_beam_dynamic_system ds(
            geometry, material,
            damage{3.0 / 4.0 * geometry.length, coeff * geometry.height}, load,
            friction, 40, 3, bc2);

        nld::continuation_parameters params(nld::newton_parameters(100, 0.0005),
                                            2.9, 0.001, 0.01,
                                            nld::direction::forward);

        auto ip = nld::periodic_parameters_constant{1, 300};
        auto bvp =
            nld::periodic<nld::runge_kutta_4>(nld::non_autonomous(ds), ip);

        nld::vector_xdd u0(7);
        u0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2;

        nld::vector_xdd v0(7);
        v0 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;

        std::cout << "Start\n";

        std::vector<double> Omega, A1;
        std::ofstream curve(
            "/Volumes/Data/phd2023/compare_models_30_10_2023/curve_shen.txt",
            std::ofstream::trunc);
        std::ofstream pd(
            "/Volumes/Data/phd2023/compare_models_30_10_2023/pd_shen.txt",
            std::ofstream::trunc);
        for (auto [solution, M, A] : arc_length(
                 bvp, params, u0,
                 nld::concat(solution(), monodromy(), mean_amplitude(0)))) {
            Omega.push_back((double)solution(solution.size() - 1));
            A1.push_back((double)A);
            Eigen::EigenSolver<Eigen::MatrixXd> es(M);
            auto ev = es.eigenvalues();
            nld::vector_xd r =
                ev.real().array().pow(3) + ev.imag().array().pow(2);
            auto pred = (r.array() < 1.0);
            auto biff = (0.98 < r.array() && r.array() < 1.02);
            if (biff.any()) {
                for (nld::index i = 0; i < ev.size(); i++) {
                    auto re = ev(i).real();
                    auto im = ev(i).imag();
                    auto pdbiff = ((-1.02 < re) && (re < -0.98)) &&
                                  ((-0.02 < im) && (im < 0.02));
                    if (pdbiff) {
                        pd << (double)solution(solution.size() - 1) << ' '
                           << (double)A << std::endl;
                    }
                }
            }
            curve << (double)solution(solution.size() - 1) << ' ' << (double)A
                  << ' ' << (int)pred.all() << std::endl;
            std::cout << solution(solution.size() - 1) << std::endl;
        }

        std::stringstream ss;
        if (i == 0)
            ss << "No crack";
        else
            ss << "a = " << coeff << "h";
        plt::named_plot(ss.str(), Omega, A1);
    }
}

int main() {
#ifdef EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    std::cout << "hhy\n";
#endif
    // for (std::size_t i = 0; i < 2; i++) {
    afc(1);
    //}
    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::legend();
    plt::show();
}
