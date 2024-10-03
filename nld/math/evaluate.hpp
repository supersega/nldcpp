#pragma once

#include <nld/core.hpp>
#include <tuple>

namespace nld::math::detail {

/// @brief Evaluate ode function at given point, time, with additional
/// arguments.
/// @details This one does not perform integration, just compute with inputs.
/// @param f Right parts of ODE.
/// @param state State variables.
/// @param t Time.
/// @param args Tuple of additional arguments.
/// @return Value of function at given inputs.
template <typename F, typename Tmp, typename S, typename Args>
auto evaluate(const F &f, const Tmp &state, S t, Args &&args) {
    return std::apply(f, std::tuple_cat(arguments(state, t), args));
}

/// @brief Evaluate implicit ode function at given point, time, with additional
/// arguments.
/// @details This one does not perform integration, just compute with inputs.
/// @param f Right parts of ODE.
/// @param state State variables.
/// @param t Time.
/// @param ydot Derivative of state variables [See: ON NUMERICAL INTEGRATION OF
/// IMPLICIT ORDINARY DIFFERENTIAL EQUATIONS]
/// @param args Tuple of additional arguments.
/// @return Value of function at given inputs.
template <typename F, typename Tmp, typename S, typename Ydot, typename Args>
auto evaluate(const F &f, const Tmp &state, S t, const Ydot &ydot,
              Args &&args) {
    return std::apply(f, std::tuple_cat(arguments(state, t, ydot), args));
}
} // namespace nld::math::detail
