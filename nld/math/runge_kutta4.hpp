#pragma once

#include <nld/core.hpp>

#include <nld/math/evaluate.hpp>
#include <nld/math/runge_kutta_parameters.hpp>

/// TODO: Make possible to pass arguments into ode solvers to make wrapping
/// elegant

namespace nld::math {

namespace detail {
/// @brief Perform one step of Runge-Kutta 4 method.
/// @param ode ODE system.
/// @param start Start time for Runge-Kutta step.
/// @param end End time for Runge-Kutta step.
/// @param initial Initial conditions at start.
/// @param args Tuple of additional arguments.
/// @returns Solution approximation at end time.
template <typename DS, Scalar S, Vector V, typename T>
V runge_kutta_4_step(const DS &ode, S start, S end, const V &initial_conditions,
                     T &&args);
} // end namespace detail

/// @brief The class to approximate ODE solution using Runge-Kutta 4 order
/// method.
/// @details This class implements standart Runge-Kutta 4 method with constant
/// step.
/// @see concepts.hpp for details of ODE Solver interface.
struct runge_kutta_4 final {
    using integration_parameters_t = constant_step_parameters;

    /// @brief Calculate solution on given interval.
    /// @details This function calculate and store whole solution on given
    /// interval.
    /// @param ode Differential equations.
    /// @param parameters Integration parameters.
    /// @param initial Initial conditions.
    /// @param args Arguments tail to ode to make call possible.
    /// @returns solution on interval.
    template <typename DS, Vector V, typename T = std::tuple<>>
    static auto solution(const DS &ode, integration_parameters_t parameters,
                         const V &initial, T &&args = no_arguments())
        -> nld::matrix_xd {
        auto [start, end, intervals] = parameters;
        auto step = (end - start) / intervals;

        auto solution_at_end = initial;

        nld::matrix_xd solution(intervals + 1, initial.size());
        solution.row(0) = initial.template cast<double>();

        for (nld::index i = 0; i < intervals; i++) {
            solution_at_end = nld::math::detail::runge_kutta_4_step(
                ode, start, start + step, solution_at_end, args);
            solution.row(i + 1) = solution_at_end.template cast<double>();
            start += step;
        }

        return solution;
    }

    /// @brief Calculate solution at end point on given interval.
    /// @details This function calculate and store end solution on given
    /// interval.
    /// @param dynamic_system Differential equations.
    /// @param constant_step_parameters Integration parameters.
    /// @param initial Initial conditions.
    /// @param args Arguments tail to ode to make call possible.
    /// @returns Solution at end point.
    template <typename DS, Vector V, typename T = std::tuple<>>
    static V end_solution(const DS &ode, integration_parameters_t parameters,
                          const V &initial, T &&args = no_arguments()) {
        auto [start, end, intervals] = parameters;
        auto step = (end - start) / intervals;

        auto interval_start = start;

        auto solution_at_end = initial;

        for (nld::index i = 0u; i < intervals; i++) {
            solution_at_end = nld::math::detail::runge_kutta_4_step(
                ode, interval_start, interval_start + step, solution_at_end,
                args);
            interval_start += step;
        }

        return solution_at_end;
    }
};

namespace detail {
template <typename DS, Scalar S, Vector V, typename T>
V runge_kutta_4_step(const DS &ode, S start, S end, const V &initial,
                     T &&args) {
    auto solution_at_end = initial;

    auto current_time = start;
    auto delta_time = end - start;
    auto time_plus_half_time_step = current_time + delta_time / 2;
    auto time_plus_time_step = current_time + delta_time;

    V tmp(solution_at_end);

    V k1 = delta_time * evaluate(ode, tmp, current_time, args);
    tmp = solution_at_end + k1 / 2;

    V k2 = delta_time * evaluate(ode, tmp, time_plus_half_time_step, args);
    tmp = solution_at_end + k2 / 2;

    V k3 = delta_time * evaluate(ode, tmp, time_plus_half_time_step, args);
    tmp = solution_at_end + k3;

    V k4 = delta_time * evaluate(ode, tmp, time_plus_time_step, args);
    solution_at_end += k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6;

    return solution_at_end;
}
} // end namespace detail
} // end namespace nld::math

namespace nld {
using nld::math::runge_kutta_4;
}
