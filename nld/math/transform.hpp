#pragma once

#include <nld/core.hpp>

namespace nld {

/// @brief Mean values in solution.
/// @param ode System of differential equation.
/// @param parameters Integration parameters.
/// @param variables Initial conditions.
/// @returns Mean values in solution.
template<typename S, typename Fn, typename P, Vector V, typename T = std::tuple<>>
auto mean(const Fn& ode, const P& parameters, const V& variables, T&& args = no_arguments()) -> V {
    auto solution = S::solution(ode, parameters, variables, std::forward<T>(args));
    return 0.5 * (solution.colwise().maxCoeff() - solution.colwise().minCoeff());
}

/// @brief Max values in solution.
/// @param ode System of differential equation.
/// @param parameters Integration parameters.
/// @param variables Initial conditions.
/// @returns Max values in solution.
template<typename S, typename Fn, typename P, Vector V, typename T = std::tuple<>>
auto max(const Fn& ode, const P& parameters, const V& variables, T&& args = no_arguments()) -> V {
    return S::solution(ode, parameters ,variables, std::forward<T>(args)).colwise().maxCoeff();
}

/// @brief Min values in solution.
/// @param ode System of differential equation.
/// @param parameters Integration parameters.
/// @param variables Initial conditions.
/// @returns Min values in solution.
template<typename S, typename Fn, typename P, Vector V, typename T = std::tuple<>>
auto min(const Fn& ode, const P& parameters, const V& variables, T&& args = no_arguments()) -> V {
    return S::solution(ode, parameters ,variables, std::forward<T>(args)).colwise().minCoeff();
}

}