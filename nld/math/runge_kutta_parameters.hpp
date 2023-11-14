#pragma once
#include <cstddef>

namespace nld::math {

/// @brief Container for ODE constant step integration parameter.
struct constant_step_parameters final {
    /// Constructor just to deduce template arguments.
    /// @param start Start integration time.
    /// @param end End integration time.
    /// @param intervals Number of integration intervals.
    constant_step_parameters(double start, double end, std::size_t intervals)
        : start{start}, end{end}, intervals{intervals} {};

    double start;          ///< start integration time.
    double end;            ///< end integration time.
    std::size_t intervals; ///< number of integration intervals.
};

/// @brief Container for ODE adaptive step integration parameter.
struct adaptive_step_parameters final {
    /// Constructor just to deduce template arguments.
    /// @param start Start integration time.
    /// @param end End integration time.
    /// @param step initial step
    /// @param error Maximal allowed error during adaptive step integration.
    adaptive_step_parameters(double start, double end, double min_step,
                             double max_step, double error)
        : start{start}, end{end}, min_step{min_step}, max_step{max_step},
          error{error} {};

    double start;    ///< start integration time.
    double end;      ///< end integration time.
    double min_step; ///< minimal step.
    double max_step; ///< maximal step.
    double error; ///< maximal allowed error during adaptive step integration.
};
} // end namespace nld::math

namespace nld {
using nld::math::adaptive_step_parameters;
using nld::math::constant_step_parameters;
} // namespace nld
