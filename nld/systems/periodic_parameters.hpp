#pragma once
#include <cstddef>
#include <nld/math.hpp>

namespace nld {
/// @brief Parameters for periodic solution continuation.
struct periodic_parameters_constant {
    explicit periodic_parameters_constant(std::size_t p, std::size_t i)
        : periods(p), intervals(i) {}

    auto to_integration_parameters() const -> nld::constant_step_parameters {
        return {0.0, 1.0 * periods, intervals};
    }

    std::size_t periods;   ///< number of periods.
    std::size_t intervals; ///< number of integration intervals.
};

/// @brief Parameters for periodic solution continuation.
struct periodic_parameters_adaptive {
    explicit periodic_parameters_adaptive(std::size_t periods, double min_step,
                                          double max_step, double error)
        : periods{periods}, min_step{min_step}, max_step{max_step},
          error{error} {}

    auto to_integration_parameters() const -> nld::adaptive_step_parameters {
        return {0.0, 1.0 * periods, min_step, max_step, error};
    }

    std::size_t periods; ///< number of periods.
    double min_step;     ///< minimal step.
    double max_step;     ///< maximal step.
    double error;        ///< maximal error.
};

} // namespace nld
