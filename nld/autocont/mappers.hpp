#pragma once

#include "nld/core/aliases.hpp"
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
    return [](const auto &fn, const auto &v) noexcept -> nld::vector_xd {
        return v.template cast<double>();
    };
}

/// @brief Function to create 2d point from solution.
/// @param n first index of variable from solution.
/// @param m second index of variable from solution.
/// @returns tuple(v[N], r)
constexpr auto point2d(nld::index n, nld::index m) noexcept {
    return [n, m](const auto &fn, const auto &v) noexcept {
        return std::tuple(static_cast<double>(v[n]), static_cast<double>(v[m]));
    };
}

/// @brief Function to get unknown variable by index.
/// @param n index of variable.
/// @returns unknown variable.
constexpr auto unknown(nld::index n) noexcept {
    return [n](const auto &fn, const auto &v) noexcept {
        return static_cast<double>(v[n]);
    };
}

/// @brief Function to request mean amplitude from periodic problem.
/// @param coordinate index of coordinate.
/// @returns mean amplitude of coordinate A.
constexpr auto mean_amplitude(nld::index coordinate) noexcept {
    return [coordinate]<typename P>(const P &periodic,
                                    const auto &variables) noexcept {
        if constexpr (nld::SimpleShootingDiscretization<P>) {
            auto [initial, args] = integration_arguments(variables, periodic);

            return static_cast<double>(mean<typename P::solver_t>(
                periodic.underlying_function(),
                periodic.integration_parameters(), initial, args)[coordinate]);
        } else if constexpr (nld::CollocationDiscretization<P>) {
            using vector_t = std::decay_t<decltype(variables)>;

            auto dim = periodic.dimension();
            vector_t u((variables.size() - 1) / dim);
            for (std::size_t i = 0; i < u.size(); ++i) {
                u[i] = variables[i * dim + coordinate];
            }

            auto max = u.maxCoeff();
            auto min = u.minCoeff();
            decltype(max) R = (max - min) / 2.0;
            return static_cast<double>(R);
        }
    };
}

/// @brief Function to request mean amplitude from periodic problem.
/// @param coordinate index of coordinate.
/// @returns mean amplitude of coordinate A.
constexpr auto half_swing(nld::index coordinate) noexcept {
    return [coordinate]<typename P>(const P &periodic,
                                    const auto &variables) noexcept {
        return std::tuple(static_cast<double>(variables(variables.size() - 1)),
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

/// @brief Mapper what generalized coordinate.
/// @returns Lambda object what generate generalized coordinate.
/// @param coordinate index of coordinate.
constexpr auto generalized_coordinate(nld::index coordinate) noexcept {
    return [coordinate]<typename P>(
               const P &periodic,
               const auto &variables) noexcept -> nld::vector_xd {
        if constexpr (nld::SimpleShootingDiscretization<P>) {
            using vector_t = std::decay_t<decltype(variables)>;

            auto [initial, args] = integration_arguments(variables, periodic);

            vector_t solution = P::solver_t::solution(
                periodic.underlying_function(),
                periodic.integration_parameters(), initial, args);

            // extract coordinate from ode solution as column vector
            return solution.col(coordinate).template cast<double>();
        } else if constexpr (nld::CollocationDiscretization<P>) {
            using vector_t = nld::vector_xd;

            auto dim = periodic.dimension();
            vector_t u((variables.size() - 1) / dim);

            // extract coordinate variables stored at next order
            // from collocation variables
            // TODO: remove node duplicates
            for (std::size_t i = 0; i < u.size(); ++i) {
                u[i] = static_cast<double>(variables[i * dim + coordinate]);
            }

            return u;
        }
    };
}

constexpr auto timestamps() noexcept {
    return []<typename P>(const P &periodic,
                          const auto &variables) noexcept -> nld::vector_xd {
        if constexpr (nld::SimpleShootingDiscretization<P>) {
            using vector_t = std::decay_t<decltype(variables)>;

            auto [initial, args] = integration_arguments(variables, periodic);

            vector_t solution = P::solver_t::solution(
                periodic.underlying_function(),
                periodic.integration_parameters(), initial, args);

            // extract time from ode solution as column vector
            return solution.col(0).template cast<double>();
        } else if constexpr (nld::CollocationDiscretization<P>) {
            using vector_t = nld::vector_xd;

            auto dim = periodic.dimension();
            const auto &mesh_parameters = periodic.mesh_parameters();
            const auto &grid = periodic.grid();
            auto N = mesh_parameters.intervals;
            auto m = mesh_parameters.collocation_points;

            vector_t t(N * m + (N - 1) + 1);

            // TODO: remove node duplicates
            for (std::size_t i = 0; i < N; ++i) {
                auto h = (grid.nodes[i + 1] - grid.nodes[i]);
                auto dt = h / m;

                for (std::size_t j = 0; j < m + 1; ++j) {
                    t[i * (m + 1) + j] = grid.nodes[i] + dt * j;
                }
            }

            return t;
        }
    };
}

} // namespace nld
