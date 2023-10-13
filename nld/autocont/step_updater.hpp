#pragma once

#include <nld/math.hpp>

namespace nld {
/// @brief Parameters for step_updator.
struct step_updater_parameters {
    double minimal_step; ///< Minimal step size.
    double maximal_step; ///< Maximal step size.
    double multipier;    ///< Step multipier.
    std::size_t limit;   ///< Minimal limit to increase step.
};

/// @brief Step updator class to calculate step for continuation algorithm.
struct step_updater final {
    /// @brief Construct a new step updater object.
    /// @param parameters configuration.
    explicit step_updater(nld::step_updater_parameters parameters) :
        good_coverage_iterations(0),
        parameters(parameters),
        current_step(parameters.minimal_step) 
    { }

    /// @brief Decrease step.
    /// @return true if we can decrease step, false otherwise.
    auto decrease_step() -> bool { 
        good_coverage_iterations = 0;
        auto aux = current_step;
        aux /= parameters.multipier;
        const bool ok = aux >= parameters.minimal_step;
        current_step = ok ? aux : current_step;
        return ok;
    }

    /// @brief Increase step iteration limit reached.
    void increase_step_if_possible() {
        good_coverage_iterations++;

        if (good_coverage_iterations > parameters.limit) {
            current_step *= parameters.multipier;
            current_step = std::min(current_step, parameters.maximal_step);
            good_coverage_iterations = 0;
        }
    }

    /// @brief Get current step.
    /// @return step. 
    double step() const { 
        return current_step;
    }
private:
    std::size_t good_coverage_iterations = 0;
    step_updater_parameters parameters;
    double current_step;
    double max_step;
};
}
