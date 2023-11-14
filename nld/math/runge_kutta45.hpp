#pragma once

#include <cmath>
#include <cstddef>
#include <nld/core.hpp>

#include <nld/math/evaluate.hpp>
#include <nld/math/runge_kutta_parameters.hpp>

#include <algorithm>

namespace nld::math {

namespace detail {
/// @brief Perform one step of Runge-Kutta 45 method.
/// @param ode ODE system.
/// @param start Start time for Runge-Kutta step.
/// @param end End time for Runge-Kutta step.
/// @param initial Initial conditions at start.
/// @param args Tuple of additional arguments.
/// @returns Solution approximation at end time.
template <typename DS, Scalar S, Vector V, typename T>
auto runge_kutta_45_step(const DS &ode, S t, S h, S h_min, S h_max, S epsilon,
                         const V &initial_conditions, T &&args)
    -> std::tuple<V, S, S, bool>;
} // end namespace detail

/// @brief The class to approximate ODE solution using Runge-Kutta 45 order
/// method with adaptive step size.
/// @details This class implements standart Runge-Kutta 4(5) method with
/// variable step.
/// @see concepts.hpp for details of ODE Solver interface.
struct runge_kutta_45 final {
    using integration_parameters_t = adaptive_step_parameters;

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
        auto [start, end, min_step, max_step, error] = parameters;
        auto intervals =
            static_cast<std::size_t>(std::floor((end - start) / max_step));
        auto step = max_step;

        auto solution_at_end = initial;

        nld::matrix_xd solution(intervals + 1, initial.size());
        solution.row(0) = initial.template cast<double>();

        auto ip = parameters;

        // TODO: end solution is not at end point
        for (nld::index i = 0; i < intervals; i++) {
            auto ip = parameters;
            ip.start = start;
            ip.end = start + step;

            solution_at_end = end_solution(ode, ip, solution_at_end, args);

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
        using nld::math::detail::runge_kutta_45_step;

        auto [start, end, h_min, h_max, epsilon] = parameters;
        auto t = start;
        auto y0 = initial;
        auto h = h_max;

        int n = 0;
        while (t < end) {
            if (t + h > end) {
                h = end - t;
            }

            auto [y, TE, h_new, ok] =
                runge_kutta_45_step(ode, t, h, h_min, h_max, epsilon, y0, args);

            if (ok) {
                n++;
                y0 = y;
                t = t + h;
            }

            h = h_new;
        }

        return y0;
    }
};

namespace detail {
template <typename DS, Scalar S, Vector V, typename T>
auto runge_kutta_45_step(const DS &ode, S t, S h, S h_min, S h_max, S epsilon,
                         const V &initial, T &&args)
    -> std::tuple<V, S, S, bool> {
    // clang-format off
    constexpr double a2 = 2.500000000000000e-01;   //  1/4
    constexpr double a3 = 3.750000000000000e-01;   //  3/8
    constexpr double a4 = 9.230769230769231e-01;   //  12/13
    constexpr double a5 = 1.000000000000000e00;    //  1
    constexpr double a6 = 5.000000000000000e-01;   //  1/2

    constexpr double b21 = 2.500000000000000e-01;  //  1/4
    constexpr double b31 = 9.375000000000000e-02;  //  3/32
    constexpr double b32 = 2.812500000000000e-01;  //  9/32
    constexpr double b41 = 8.793809740555303e-01;  //  1932/2197
    constexpr double b42 = -3.277196176604461e00;  // -7200/2197
    constexpr double b43 = 3.320892125625853e00;   //  7296/2197
    constexpr double b51 = 2.032407407407407e00;   //  439/216
    constexpr double b52 = -8.000000000000000e00;  // -8
    constexpr double b53 = 7.173489278752436e00;   //  3680/513
    constexpr double b54 = -2.058966861598441e-01; // -845/4104
    constexpr double b61 = -2.962962962962963e-01; // -8/27
    constexpr double b62 = 2.000000000000000e00;   //  2
    constexpr double b63 = -1.381676413255361e00;  // -3544/2565
    constexpr double b64 = 4.529727095516569e-01;  //  1859/4104
    constexpr double b65 = -2.750000000000000e-01; // -11/40

    constexpr double r1 = 2.777777777777778e-03;   //  1/360
    constexpr double r3 = -2.994152046783626e-02;  // -128/4275
    constexpr double r4 = -2.919989367357789e-02;  // -2197/75240
    constexpr double r5 = 2.000000000000000e-02;   //  1/50
    constexpr double r6 = 3.636363636363636e-02;   //  2/55

    constexpr double c1 = 1.157407407407407e-01;   //  25/216
    constexpr double c3 = 5.489278752436647e-01;   //  1408/2565
    constexpr double c4 = 5.353313840155945e-01;   //  2197/4104
    constexpr double c5 = -2.000000000000000e-01;  // -1/50.0

    constexpr double d1 = 16.0 / 135.0;
    constexpr double d3 = 6656.0 / 12825.0;
    constexpr double d4 = 28561 / 56430.0;
    constexpr double d5 = -9.0 / 50.0;
    constexpr double d6 = 2.0 / 55.0;

    auto y = initial;

    V tmp(y);

    V k1 = h * evaluate(ode, tmp, t, args);
    tmp = y + b21 * k1;

    V k2 = h * evaluate(ode, tmp, t + a2 * h, args);
    tmp = y + b31 * k1 + b32 * k2;

    V k3 = h * evaluate(ode, tmp, t + a3 * h, args);
    tmp = y + b41 * k1 + b42 * k2 + b43 * k3;

    V k4 = h * evaluate(ode, tmp, t + a4 * h, args);
    tmp = y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4;

    V k5 = h * evaluate(ode, tmp, t + a5 * h, args);
    tmp = y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5;

    V k6 = h * evaluate(ode, tmp, t + a6 * h, args);

    V z = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5;
    y   = y + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6;
    
    V TEv = z - y;
    TEv = TEv.cwiseAbs();
    S TE = (S)TEv.maxCoeff();

    S s = 0.84 * pow(epsilon * h / TE, 0.25);

    S h_new = h * std::min(std::max(s, 0.1), 4.0);
    S h_clamped = std::clamp(h_new, h_min, h_max);

    // clang-format on
    return std::tuple(y, TE, h_clamped, !(h_new > h_min && TE > epsilon));
}
} // end namespace detail
} // end namespace nld::math

namespace nld {
using nld::math::runge_kutta_45;
}
