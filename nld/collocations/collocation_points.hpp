#pragma once

#include "nld/math/segment.hpp"
#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld::collocations {
/// @brief Concept for collocation points builder
/// @details The builder must be a function that receives a segment and the
/// number of collocation points and returns a vector with the collocation
/// points (tau_0 != , tau_1, ..., tau_m)
template <typename F>
concept CollocationPointsBuilder = requires(F f) {
    {
        f(std::declval<nld::segment>(), std::declval<std::size_t>())
    } -> std::same_as<nld::vector_xd>;
};

namespace detail {
/// @brief Create Legandre collocation points
/// @details N(number of intervals) = nodes.size() - 1
/// @param nodes Mesh nodes [t0, t1, ..., tN]
/// @param colloquation_points Number of collocation points per interval
/// @return Vector with collocation points
template <CollocationPointsBuilder F>
auto collocation_points(const nld::vector_xd &nodes,
                        std::size_t colloquation_points, const F &builder)
    -> nld::vector_xd {
    // The collocation points are next:
    // [t0, t0_1/m, t0_2/m, t1, ...]
    //    |  collocation  | ...
    auto N = nodes.size() - 1;
    auto m = colloquation_points;
    nld::vector_xd points(N * m);

    for (std::size_t j = 1; j < N + 1; ++j) {
        nld::segment interval{nodes[j - 1], nodes[j]};
        auto L = builder(interval, m);
        for (std::size_t i = 0; i < m; ++i) {
            std::size_t index = (j - 1) * m + i;
            points[index] = L[i];
        }
    }

    return points;
}
} // namespace detail
} // namespace nld::collocations
