#include <iostream>
#include <nld/calculus.hpp>
#include <nld/math.hpp>

#include <matplotlibcpp.h>

#include "beam_common.hpp"

namespace plt = matplotlibcpp;

struct linear_parametric_system {
    linear_parametric_system(nld::mechanics::beam const &b,
                             nld::mechanics::force const &f, int dofs)
        : geometry(b.geometry), material(b.material), force(f), dofs(dofs),
          bc(b.geometry.length) {}

    auto calculate_matricies() {
        auto rho = material.density;
        auto A = geometry.area();
        auto I = geometry.inertia();
        auto E = material.young_modulus;
        auto l = geometry.length;
        auto F = force.amplitude;
        Fcr = M_PI * M_PI * E * I / (l * l);

        auto domain = nld::constant_domain{0.0, l};
        constexpr auto Nb = 90;
        auto bd = nld::basis_definition{Nb, {0.0, l}};
        auto bsb = nld::bspline_3_basis(bd);
        auto sp = nld::space<1>{};
        auto &x = sp.coords()[0];
        auto tf = nld::test_functions(sp, bsb);
        auto u = bc * tf;

        nld::tensor_expression_calculator<nld::cquad> calculator(
            nld::cquad::integration_options{1.0e-8, 1.0e-8, 300});

        auto kinetic_energy_form = nld::integral(rho * A * u * u, domain);
        auto potential_energy_bending_form = nld::integral(
            E * I * nld::laplacian(u) * nld::laplacian(u), domain);

        auto epnc = nld::solve_eigenvalue_problem<nld::cquad>(
            kinetic_energy_form, potential_energy_bending_form,
            nld::cquad::integration_options{1.0e-8, 1.0e-11, 300});
        auto ef = epnc.eigenvalues.cwiseSqrt();
        decltype(auto) eigenvectors = sqrt(rho * A) * epnc.eigenvectors;

        std::cout << "Eigenfrequencies: " << ef << std::endl;

        auto first_frequency_analytical =
            M_PI * M_PI * sqrt(E * I / (rho * A)) / (l * l);
        std::cout << "First frequency analytical: "
                  << first_frequency_analytical << std::endl;

        auto omega_0 = ef(0);

        auto U = nld::eigenfunctions(u, std::move(eigenvectors), dofs);
        auto T = nld::integral(rho * A * U * U, domain);
        auto [M] = calculator.calculate(T);
        auto K_bi =
            nld::integral(nld::laplacian(U) * nld::laplacian(U), domain);
        auto [K_b] = calculator.calculate(K_bi);
        auto K_si = nld::integral(
            nld::diff(U, nld::wrt(x)) * nld::diff(U, nld::wrt(x)), domain);
        auto [K_s] = calculator.calculate(K_si);

        auto integrator = nld::cquad{};
        auto inner_domain = nld::variable_domain{[](auto x) { return 0.0; },
                                                 [l](auto x) { return l; }};
        auto inner_integral = nld::variable_integral(
            U * U, integrator, inner_domain,
            nld::cquad::integration_options{1.0e-4, 1.0e-4, 300});

        auto Y_ = nld::integral(inner_integral, domain);
        auto [Y_s] = calculator.calculate(Y_);

        Eigen::Map<Eigen::MatrixXd> M_(M.data(), dofs, dofs);
        Eigen::Map<Eigen::MatrixXd> K_0(K_s.data(), dofs, dofs);
        Eigen::Map<Eigen::MatrixXd> K_1(K_b.data(), dofs, dofs);
        Eigen::Map<Eigen::MatrixXd> Y(Y_s.data(), dofs, dofs);

        Eigen::MatrixXd K_0_hat = M_.inverse() * K_0 * Fcr / omega_0 / omega_0;
        Eigen::MatrixXd K_1_hat =
            M_.inverse() * E * I * K_1 / omega_0 / omega_0;

        std::cout << "Mass matrix: \n" << M_ << std::endl;
        std::cout << "Str stiffness matrix: \n" << K_0 << std::endl;
        std::cout << "Bending stiffness matrix: \n" << K_1 << std::endl;
        std::cout << "Str stiffness matrix hat: \n" << K_0_hat << std::endl;
        std::cout << "Bending stiffness matrix hat: \n" << K_1_hat << std::endl;
        std::cout << "Y matrix: \n" << Y << std::endl;
    }

    void print() { std::cout << "Fcr = " << Fcr << std::endl; }

    nld::mechanics::geometry geometry;
    nld::mechanics::material material;
    nld::mechanics::force force;
    int dofs;
    nld::mechanics::hinged_beam_bc bc;

    double Fcr; // critical nld::mechanics::force
};

auto gsl_error_handler(const char *reason, const char *file, int line,
                       int gsl_errno) -> void {
    std::cerr << "Error: " << reason << " at " << file << ":" << line
              << " error code: " << gsl_errno << std::endl;
}

int main() {
    gsl_set_error_handler(&gsl_error_handler);
    nld::mechanics::material p{2.013e11, 7803};
    nld::mechanics::geometry g{558.7e-3, 5.0e-3, 11.95e-3};
    nld::mechanics::force f{1.0};
    nld::mechanics::beam b{g, p};
    linear_parametric_system lps(b, f, 5);
    lps.calculate_matricies();
    lps.print();
}
