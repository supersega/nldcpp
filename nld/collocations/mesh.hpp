#pragma once

#include <nld/collocations/collocation_points.hpp>
#include <nld/collocations/nodes.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld::collocations {

/// @brief parameters for mesh
struct mesh_parameters final {
    std::size_t intervals;
    std::size_t collocation_points;
};

/// @brief Collocation mesh with nodes and collocation points
struct mesh final {
    /// @brief Create a mesh
    /// @param parameters Mesh parameters
    /// @param nodes_builder Function that creates the nodes
    /// @param collocation_points_builder Function that creates the collocation
    /// points on each interval
    /// @details The nodes vector has size N + 1, where N is the number of
    /// intervals. The collocation points vector has size N * m, where m is The
    /// number of collocation points per interval.
    template <NodesBuilder F, CollocationPointsBuilder G>
    explicit mesh(const mesh_parameters &parameters, const F &nodes_builder,
                  const G &collocation_points_builder)
        : nodes{nodes_builder(parameters.intervals)},
          collocation_points{
              detail::collocation_points(nodes, parameters.collocation_points,
                                         collocation_points_builder)} {}
    nld::vector_xd nodes;              ///< Mesh nodes
    nld::vector_xd collocation_points; ///< Collocation points
};

} // namespace nld::collocations
