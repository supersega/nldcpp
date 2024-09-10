#pragma once

#include <cstddef>

namespace nld::math {
/// @brief Structure which holds Newton method parameters.
struct newton_parameters final {
    /// @brief constructor just to deduce template arguments
    /// @param max_iterations maximal iterations
    /// @param tolerance evaluation tolerance (should be floating point)
    newton_parameters(std::size_t max_iterations, double tolerance)
        : max_iterations{max_iterations}, tolerance{tolerance} {}

    std::size_t max_iterations; ///< maximal iterations.
    double tolerance;           ///< evaluation tolerance.
};

/// @brief Structure which holds info about Newton method solution.
struct newton_info final {
    /// @brief Explicit convert to bool.
    /// @return true if solution convergence.
    explicit operator bool() const { return is_convergence; }

    std::size_t number_of_done_iterations; ///< Number of done iterations.
    bool is_convergence;                   ///< Convergence flag.
};

} // end namespace nld::math

namespace nld {
using nld::math::newton_info;
using nld::math::newton_parameters;
} // namespace nld
