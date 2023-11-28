#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld::collocations {
/// @brief Concept for nodes builder
/// @details The builder must be a function that receives the number of
/// intervals and returns a vector with the nodes (t0 = 0, t1, ..., tN = 1)
template <typename F>
concept NodesBuilder = requires(F f) {
    { f(std::declval<std::size_t>()) } -> std::same_as<nld::vector_xd>;
};

/// @brief Create a uniform mesh nodes
/// @param intervals Number of intervals (N)
/// @return Vector with mesh nodes [t0 = 0, t1, ..., tN = 1]
inline nld::vector_xd uniform_mesh_nodes(std::size_t intervals) {
    auto N = intervals;
    nld::vector_xd nodes(N + 1);
    double step = 1.0 / N;
    for (int i = 0; i < N + 1; ++i) {
        nodes[i] = i * step;
    }
    return nodes;
}
} // namespace nld::collocations
