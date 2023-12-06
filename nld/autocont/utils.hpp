#pragma once

#include <cstddef>

namespace nld::detail {
/// @brief Compute new step.
/// @param done_iterations number of done iterations.
/// @param step current step.
/// @param min_step minimal step size.
/// @param max_step maximal step size.
template <typename T>
T updated_step(std::size_t done_iterations, T step, T min_step, T max_step) {
    constexpr auto iterations_upper_border = 6;
    constexpr auto iterations_lower_border = 2;
    constexpr auto step_coefficient = 1.667;

    if (auto new_step = step * step_coefficient;
        done_iterations <= iterations_lower_border && new_step < max_step)
        return new_step;

    if (auto new_step = step / step_coefficient;
        done_iterations >= iterations_upper_border && new_step > min_step)
        return new_step;

    return step;
}
} // namespace nld::detail
