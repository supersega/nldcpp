#pragma once

#include <nld/autocont/problem.hpp>
#include <nld/autocont/systems.hpp>

namespace nld {

/// @brief Base periodic problem class.
/// @tparam T Type.
template <typename T>
struct periodic_base;

/// @brief If dynamic system is autonomous then we have to know previous solution.
/// @tparam Fn Generic function type.
template <typename Fn>
struct periodic_base<nld::autonomous<Fn>> : nld::problem<nld::autonomous<Fn>> {
    using function_t = nld::autonomous<Fn>; 
    using vector_t = typename function_t::vector_t;
    using problem_t = nld::problem<function_t>;

    using problem_t::problem_t;

    /// @brief Set previous solution
    template<typename Vector>
    void set_previous_solution(const Vector& state) {
        previous_step_solution = state;
    }

    /// @brief Get previous solution
    auto previous_solution() const -> const vector_t& {
        return previous_step_solution;
    }
protected:
    vector_t previous_step_solution;
};

/// @brief If dynamic system is non autonomous then we don't have to know previous solution.
/// @tparam Fn Generic function type.
template <typename Fn>
struct periodic_base<nld::non_autonomous<Fn>> : nld::problem<nld::non_autonomous<Fn>> {
    using function_t = nld::non_autonomous<Fn>;
    using problem_t = nld::problem<function_t>;
    
    using problem_t::problem_t;
};
}
