#pragma once

#include <nld/core.hpp>

#include <nld/calculus/tensor_expression_calculator.hpp>

namespace nld {

/// @brief The eigenvalue problem solution.
struct eigenvalue_problem_solution final {
    nld::vector_xd eigenvalues;  ///< The eigenvalues vector.
    nld::matrix_xd eigenvectors; ///< The eigenvectors matrix.
};

/// @brief Solve eigenvalue problem with deduced types.
/// @param kinetic_energy The kinetic energy expression.
/// @param potential_energy The potential energy expression.
/// @param domain The integration domain.
/// @param options The integration options.
/// @return eigenvalue_problem<I, K, P> 
template<typename I, typename K, typename P>
auto solve_eigenvalue_problem(K&& kinetic_energy, P&& potential_energy, nld::integration_options options = integration_options{ }) -> eigenvalue_problem_solution {
    nld::tensor_expression_calculator<I> calculator(options);

    auto [mass_tensor] = calculator.calculate(kinetic_energy);
    auto [stiffness_tensor] = calculator.calculate(potential_energy);

    auto dimensions = mass_tensor.dimensions();

    Eigen::Map<nld::matrix_xd> mass(mass_tensor.data(), dimensions[0], dimensions[1]);
    Eigen::Map<nld::matrix_xd> stiffness(stiffness_tensor.data(), dimensions[0], dimensions[1]);

    Eigen::GeneralizedSelfAdjointEigenSolver<nld::matrix_xd> es(stiffness, mass);
    return eigenvalue_problem_solution { es.eigenvalues(), es.eigenvectors() };
}
}