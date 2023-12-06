#pragma once

#include <nld/systems/jacobian_mixin.hpp>
#include <nld/systems/problem.hpp>

namespace nld {

/// @brief Equilibrium solution for autonomous ode problem.
/// @details This class describes equilibrium problem for
/// autonomous ode system. The jacobian_mixin used to
/// inject simple jacobian that just evaluated using operator().
/// Example of usage:
/// @code
/// // The Fn type in equilibrium should have next signature
/// // x - unknowns
/// // lambda - continuation parameter
/// auto problem(nld::vector_xdd& x, dual lambda) -> nld::vector_xdd;
///
/// int main() {
///     nld::vector_xdd y = ...; ///< Vector of size x.size() + 1
///
///     auto eq = nld::equilibrium(problem);
///     auto val = eq(y); // compute equilibrium problem at 'y'
/// }
/// @endcode
/// @see jacobian_mixin.hpp
template <typename Fn>
struct equilibrium final : nld::problem<Fn>,
                           nld::jacobian_mixin<equilibrium<Fn>> {
    using problem_t = problem<Fn>;
    using vector_t = decltype(std::declval<Fn>()(nld::utils::any_type{},
                                                 nld::utils::any_type{}));

    /// @brief Construct equilibrium problem from function
    /// @param fn Nonlinear function.
    explicit equilibrium(Fn &&fn) : problem_t(std::forward<Fn>(fn)) {}

    /// @brief Compute equilibrium at given point.
    /// @details Equilibrium for autonomous ODE y' = f(y(t), lambda).
    /// @param variables unknowns.
    template <typename Vector>
    auto operator()(const Vector &variables) const -> Vector {
        const auto dim = variables.size();

        auto parameter = variables(dim - 1);

        return this->function(variables.head(dim - 1), parameter);
    }
};

template <typename F>
equilibrium(F &&) -> equilibrium<F>;

} // namespace nld
