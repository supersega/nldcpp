#pragma once

#include <nld/autocont/arc_length_raw.hpp>
#include <nld/autocont/mappers.hpp>
#include <nld/core/aliases.hpp>
#include <nld/systems/dimension.hpp>
#include <nld/systems/equilibrium.hpp>
#include <nld/systems/homotopy.hpp>
#include <nld/systems/ode.hpp>
#include <nld/systems/periodic.hpp>

namespace nld {

/// @brief First guess for continuation.
struct first_guess final {
    nld::vector_xdd point;      ///< First point for continuation.
    nld::vector_xdd tangential; ///< First tangent for continuation.
};

/// @brief Compute data for continuation.
template <typename F>
first_guess evaluate_firs_guess(const F &f) = delete;

/// @brief Compute data for continuation. Evaluate firs guess for equilibrium.
template <typename Fn>
first_guess evaluate_firs_guess(const equilibrium<Fn> &eq, dimension size) {
    const index dimension = size;
    const auto unknowns = dimension + 1;
    dual parameter = 0.1;

    auto system = [&eq, dimension, parameter](const auto &u) {
        nld::vector_xdd uf(dimension + 1);
        uf.head(dimension) = u;
        // bind parameter to value
        uf(dimension) = parameter;

        return eq(uf);
    };

    nld::vector_xdd known = nld::vector_xdd::Zero(unknowns);

    auto fixed_point_homotopy = [](const auto &x) -> nld::vector_xdd {
        return -x;
    };

    convex_homotopy homotopy(std::move(system),
                             std::move(fixed_point_homotopy));

    nld::vector_xdd unknown(unknowns);
    unknown << nld::vector_xdd::Zero(dimension), parameter;

    nld::vector_xdd tangential = nld::vector_xdd::Zero(unknowns);
    tangential(dimension) = 1.0;

    continuation_parameters params(newton_parameters(25, 0.00009), 8.5, 0.002,
                                   0.0025, direction::forward);

    for (auto value : nld::internal::arc_length_raw(
             std::move(homotopy), params, known, tangential, solution())) {
        auto kappa = value(dimension);
        if (abs(kappa - 1.0) < 1.0e-3) {
            unknown.head(dimension) = value.head(dimension);
            break;
        }
    }

    nld::vector_xdd tan = nld::vector_xdd::Zero(unknowns);

    tan(dimension) = 1.0;

    return first_guess{unknown, tan};
}

/// @brief Compute data for continuation. Evaluate firs guess for periodic.
// template<template<typename...> class S, typename Fn>
// first_guess evaluate_firs_guess(const periodic<S, Fn>& pd) {
//     if constexpr (is_autonomous_v<periodic<S, Fn>::dynamic_system_t>) {
//         const auto dimension = pd.dimension();
//         const auto unknowns = dimension + 2;

//         nld::vector_xdd tangential = nld::vector_xdd::Zero(unknowns);
//         tangential(unknowns - 1) = 1.0;

//         return first_guess { nld::vector_xdd::Zero(unknowns),
//         std::move(tangential) };
//     }
//     else {
//         const auto dimension = pd.dimension();
//         const auto unknowns = dimension + 1;

//         nld::vector_xdd tangential = nld::vector_xdd::Zero(unknowns);
//         tangential(unknowns - 1) = 1.0;

//         return first_guess { nld::vector_xdd::Zero(unknowns),
//         std::move(tangential) };
//     }
// }

} // namespace nld
