#include <catch.hpp>

#include <cmath>
#include <iostream>
using namespace std;

#include <nld/calculus.hpp>
using namespace nld;

struct hinged_beam_bc : boundary_condition {
    explicit hinged_beam_bc(double l) : length(l) {}

    auto value() const {
        return [l = length](auto x) -> adnum { return x * (x - l); };
    }

private:
    double length;
};

TEST_CASE("Hinged beam eigenvalue problem ") {
    auto length = 0.177;
    auto height = 0.01;
    auto width = 0.01;
    auto rho = 7800;
    auto A = height * width;
    auto E = 2.1e11;
    auto d = height / 2.0;
    auto b = width / 2.0;
    auto I = 4.0 / 3.0 * b * d * d * d;
    auto l = length;

    auto domain = segment{0.0, length};
    auto bd = basis_definition{40, {0.0, length}};
    auto bsb = bspline_3_basis(bd);
    auto sp = space<1>{};
    auto tf = test_functions(sp, bsb);
    auto bc = hinged_beam_bc(length);
    auto u = bc * tf;
    auto a = 0.4 * d;
    auto g = (a / d);
    auto m = 1 / (1.0 + 0.75 * g * g - 1.5 * g - 1.8 * g * g * g);
    std::cout << "m = " << m << std::endl;
    auto alpha = 1.276;
    auto xc = 0.5 * l;
    auto L = [=](auto x) -> adnum {
        return (m - 1.0) * exp(-2.0 * alpha * abs(x - xc) / d);
    };
    auto Q = scalar_function([=](adnum x) -> adnum { return 1.0; });
    auto kinetic_energy = nld::integral(rho * A * u * u, domain);
    auto potential_energy =
        nld::integral(E * I * Q * laplacian(u) * laplacian(u), domain);
    auto shift = nld::integral(u * delta_function(0.6), domain);

    auto ep = solve_eigenvalue_problem<gauss_kronrod21>(
        kinetic_energy, potential_energy,
        gauss_kronrod21::integration_options{1.0e-6, 1.0e-6, 200});
    auto ef = ep.eigenvalues.cwiseSqrt();
    auto eigenvectors = ep.eigenvectors;

    auto U = eigenfunctions(u, std::move(eigenvectors), 2);
    auto &x = sp.coords()[0];
    auto nl_term = nld::integral(
        nld::integral(diff(U, autodiff::wrt(x)) * diff(U, autodiff::wrt(x)),
                      domain) *
            diff(U, autodiff::wrt(x)) * diff(U, autodiff::wrt(x)),
        domain);

    // auto TK = rho * A * U * U;
    // auto PK = E * I * Q * laplacian(U) * laplacian(U);
    // //E * I * nld::integral(Q * diff(U, x) * diff(U, x));

    tensor_expression_calculator<gauss_kronrod21> calculator(
        gauss_kronrod21::integration_options{1.0e-5, 1.0e-5, 300});
    auto [M] = calculator.calculate(nl_term);
    std::cout << M;
    REQUIRE(1 == 2);
}
