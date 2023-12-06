#pragma once

#include <nld/math.hpp>

#include <nld/systems/periodic.hpp>

namespace nld {

/// @brief Concatenate map functions.
/// @param map map functions.
/// @returns function which creates tuple of maps results.
template <typename... M>
constexpr auto concat(M &&...map) noexcept {
    return [... map = std::forward<M>(map)](const auto &fn,
                                            const auto &v) noexcept {
        return std::tuple(map(fn, v)...);
    };
};

/// @brief Function to create map which returns variables.
/// @returns solution of nonlinear system.
constexpr auto solution() {
    return [](const auto &fn, const auto &v) noexcept { return v; };
}

/// @brief Function to create 2d point from solution.
/// @param n first index of variable from solution.
/// @param m second index of variable from solution.
/// @returns tuple(v[N], r)
constexpr auto point2d(nld::index n, nld::index m) noexcept {
    return [n, m](const auto &fn, const auto &v) noexcept {
        return std::tuple(v[n], v[m]);
    };
}

/// @brief Function to request mean amplitude from periodic problem.
/// @param coordinate index of coordinate.
/// @returns mean amplitude of coordinate A.
constexpr auto mean_amplitude(nld::index coordinate) noexcept {
    return [coordinate]<typename P>(const P &periodic,
                                    const auto &variables) noexcept {
        auto [initial, args] = integration_arguments(variables, periodic);

        return mean<typename P::solver_t>(periodic.underlying_function(),
                                          periodic.integration_parameters(),
                                          initial, args)[coordinate];
    };
}

/// @brief Function to request mean amplitude from periodic problem.
/// @param coordinate index of coordinate.
/// @returns mean amplitude of coordinate A.
constexpr auto half_swing(nld::index coordinate) noexcept {
    return [coordinate]<typename P>(const P &periodic,
                                    const auto &variables) noexcept {
        return std::tuple(variables(variables.size() - 1),
                          mean_amplitude(coordinate)(periodic, variables));
    };
}

/// @brief Mapper what evaluates monodromy matrix for periodic .
/// @returns Lambda object what evaluates monodromy Matrix.
constexpr auto monodromy() noexcept {
    return []<typename P>(const P &periodic, auto variables) noexcept {
        const auto dimension =
            variables.size() - periodic.non_state_variables();

        decltype(periodic(variables)) result(variables.size());
        auto jacobian = periodic.jacobian(wrt(variables.head(dimension)),
                                          at(variables), result);
        jacobian += nld::matrix_xd::Identity(dimension, dimension);

        return jacobian;
    };
}
} // namespace nld
