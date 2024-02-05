#pragma once

#include <nld/systems/ode.hpp>
#include <nld/systems/problem.hpp>

namespace nld {

namespace internal {

template <typename F>
struct with_previous_solution {
    using function_t = std::decay_t<F>;
    using vector_t = typename function_t::vector_t;

    /// @brief Set previous solution
    template <typename Vector>
    void set_previous_solution(const Vector &state) {
        previous_step_solution = state;
    }

    /// @brief Get previous solution
    auto previous_solution() const -> const vector_t & {
        return previous_step_solution;
    }

protected:
    vector_t previous_step_solution;
};

template <typename F>
struct without_previous_solution {};

} // namespace internal

/// @brief Base periodic problem class.
/// @tparam T Type.
template <typename T>
struct periodic_base
    : nld::problem<T>,
      std::conditional_t<nld::is_autonomous_v<std::decay_t<T>>,
                         internal::with_previous_solution<T>,
                         internal::without_previous_solution<T>> {
    using problem_t = nld::problem<T>;

    using problem_t::problem_t;
};
} // namespace nld
