#pragma once

#include "nld/core/aliases.hpp"
#include <nld/collocations/mesh.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>

#include <autodiff/forward.hpp>

namespace nld::collocations {

/// @brief Evaluate basis and derivatives for a given element in collocation
/// points
/// @tparam F Basis builder
template <typename F>
struct collocation_on_elements final {
    /// @brief Create a collocation derivatives object
    /// @param grid Collocations mesh
    /// @param basis_builder Basis builder
    explicit collocation_on_elements(const mesh &grid, F &&basis_builder)
        : collocations_mesh{grid},
          basis_builder{std::forward<F>(basis_builder)} {}

    /// @brief Evaluate basis in mesh node
    /// @param in Mesh node index, 1 <= in <= N - 1
    /// @return Matrix with basis values at node for left and right elements
    auto values_in_mesh_node(std::size_t in) -> nld::matrix_xd {
        auto N = collocations_mesh.nodes.size() - 1;
        auto M = collocations_mesh.collocation_points.size();
        std::size_t m = M / N;

        nld::matrix_xd result(m + 1, 2);
        for (std::size_t i = 0; i < 2; ++i) {

            nld::segment interval{collocations_mesh.nodes[in - 1 + i],
                                  collocations_mesh.nodes[in + i]};
            auto basis = basis_builder(interval, m);

            double t = collocations_mesh.nodes[in];
            for (std::size_t j = 0; j < m + 1; ++j) {
                result(j, i) = basis(j, t);
            }
        }

        return result;
    }

    /// @brief Evaluate basis for given Element
    /// @param ie Element index
    /// @return Basis matrix of size [m x m + 1]
    auto values_in_collocation_points(std::size_t ie) const -> nld::matrix_xd {
        auto N = collocations_mesh.nodes.size() - 1;
        auto M = collocations_mesh.collocation_points.size();

        std::size_t m = M / N;
        std::size_t shift = ie * m;
        nld::segment interval{collocations_mesh.nodes[ie],
                              collocations_mesh.nodes[ie + 1]};
        auto basis = basis_builder(interval, m);

        nld::matrix_xd result(m + 1, m);
        for (std::size_t i = 0; i < m; ++i) {
            double t = collocations_mesh.collocation_points[shift + i];
            for (std::size_t j = 0; j < m + 1; ++j) {
                result(j, i) = basis(j, t);
            }
        }

        return result;
    }

    /// @brief Evaluate derivatives matrix of basis for given element
    /// @param ie Element index
    /// @return Derivatives matrix of size [m x m + 1]
    auto derivatives_in_collocation_points(std::size_t ie) const
        -> nld::matrix_xd {
        auto N = collocations_mesh.nodes.size() - 1;
        auto M = collocations_mesh.collocation_points.size();

        std::size_t m = M / N;
        std::size_t shift = ie * m;
        nld::segment interval{collocations_mesh.nodes[ie],
                              collocations_mesh.nodes[ie + 1]};
        auto basis = basis_builder(interval, m);

        nld::matrix_xd result(m + 1, m);
        for (std::size_t i = 0; i < m; ++i) {
            nld::dual t = collocations_mesh.collocation_points[shift + i];
            for (std::size_t j = 0; j < m + 1; ++j) {
                using autodiff::forward::at;
                using autodiff::forward::wrt;

                auto dldt = autodiff::derivative(basis, wrt(t), at(j, t));
                result(j, i) = dldt;
            }
        }

        return result;
    }

private:
    const mesh &collocations_mesh; ///< Collocations mesh
    F basis_builder;               ///< Basis builder
};

template <typename F>
collocation_on_elements(const mesh &, F &&) -> collocation_on_elements<F>;

} // namespace nld::collocations
