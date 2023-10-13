#pragma once

namespace nld {

/// @brief Integration options.
struct integration_options {
    double absolute_tolerance = 1.49e-8; ///< Absolute integration tolerance.
    double relative_tolerance = 1.49e-8; ///< Relative integration tolerance.
    std::size_t max_iterations = 50;     ///< Limit of iterations.
};
}
