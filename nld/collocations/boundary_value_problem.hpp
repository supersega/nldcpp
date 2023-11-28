#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/collocations/basis_builder.hpp>
#include <nld/collocations/collocation_on_elements.hpp>
#include <nld/collocations/lagrange_basis.hpp>
#include <nld/collocations/legandre_collocation_points.hpp>
#include <nld/collocations/mesh.hpp>
#include <utility>

namespace nld::collocations {
/// @brief Boundary value problem
/// @tparam F Type of the function
/// @tparam Bc Type of the boundary conditions
/// @tparam Basis Type of the basis builder
/// @details This class represents a boundary value problem of the form
/// \f[
///    \begin{cases}
///    \mathbf{u}'(t) = \mathbf{f}(\mathbf{u}(t), t) \\
///    \mathbf{b}(\mathbf{u}(t_0), \mathbf{u}(t_N)) = 0 \\
///    \end{cases}
/// \f]
template <typename F, typename Bc, typename Basis>
struct boundary_value_problem final {
    /// @brief Create a boundary value problem
    /// @param f Function of the boundary value problem
    /// @param bc Boundary conditions of the boundary value problem
    /// @param basis Basis builder of the boundary value problem
    /// @param parameters Mesh parameters of the boundary value problem
    /// @param dimension Dimension of the boundary value problem
    explicit boundary_value_problem(
        F &&f, Bc &&bc, Basis &&basis,
        nld::collocations::mesh_parameters parameters, std::size_t dimension)
        : function(std::forward<F>(f)),
          boundary_conditions(std::forward<Bc>(bc)),
          basis_builder(std::forward<Basis>(basis)), parameters(parameters),
          dimension(dimension),
          grid(nld::collocations::mesh(
              parameters, nld::collocations::uniform_mesh_nodes,
              nld::collocations::legandre_collocation_points)) {}

    /// @brief Evaluate the jacobian of the boundary value problem
    /// @tparam Wrt Type of the variables to differentiate with respect to
    /// @tparam At Type of the variables to evaluate at
    /// @tparam V Type of the result of the function
    /// @param wrt Variables to differentiate with respect to
    /// @param at Variables to evaluate at
    /// @param v Value of the function
    template <typename Wrt, typename At, typename V>
    auto jacobian(Wrt &&wrt, At &&at, V &&v) const -> nld::matrix_xd {
        auto jac = autodiff::forward::jacobian(*this, wrt, at, v);
        return jac;
    }

    /// @brief Evaluate the boundary value problem
    /// @tparam U Type of the variables to evaluate at
    /// @param u Variables to evaluate at
    /// @return Value of the boundary value problem
    auto operator()(const nld::vector_xdd &u) const -> nld::vector_xdd {
        nld::collocations::collocation_on_elements evaluator(grid,
                                                             basis_builder);

        auto N = parameters.intervals;
        auto m = parameters.collocation_points;
        std::size_t n = dimension;

        nld::vector_xdd f(N * n * m + (N - 1) * n + n);
        f = nld::vector_xdd::Zero(N * n * m + (N - 1) * n + n);

        for (std::size_t j = 0; j < N; ++j) {
            auto values = evaluator.values_in_collocation_points(j);
            auto derivatives = evaluator.derivatives_in_collocation_points(j);

            for (std::size_t k = 0; k < m; ++k) {
                auto t = grid.collocation_points[j * m + k];

                nld::vector_xdd p = nld::vector_xdd::Zero(n);
                nld::vector_xdd p_prime = nld::vector_xdd::Zero(n);
                for (std::size_t l = 0; l < m + 1; ++l) {
                    for (std::size_t i = 0; i < n; ++i) {
                        auto q = u[j * (m + 1) * n + l * n + i];
                        p[i] += values(k, l) * q;
                        p_prime[i] += derivatives(k, l) * q;
                    }
                }

                auto f_p = function(p, t);
                f.segment(j * (m + 1) * n + k * n, n) = p_prime - f_p;
            }

            if (j < N - 1) {
                auto values_in_mesh_node = evaluator.values_in_mesh_node(j + 1);

                nld::vector_xdd continuity = nld::vector_xdd::Zero(n);
                for (std::size_t l = 0; l < m + 1; ++l) {
                    for (std::size_t i = 0; i < n; ++i) {
                        auto ql = u[j * (m + 1) * n + l * n + i];
                        auto qr = u[(j + 1) * (m + 1) * n + l * n + i];
                        continuity[i] += values_in_mesh_node(0, l) * ql -
                                         values_in_mesh_node(1, l) * qr;
                    }
                }

                f.segment(j * (m + 1) * n + m * n, n) = continuity;
            }
        }

        f.segment(N * m * n + (N - 1) * n, n) =
            boundary_conditions(u.head(n), u.tail(n));

        return f;
    }

private:
    F function;                 ///< Function of the boundary value problem
    Bc boundary_conditions;     ///< Boundary conditions of the boundary value
                                ///< problem
    Basis basis_builder;        ///< Basis builder of the boundary value problem
    mesh_parameters parameters; ///< Mesh parameters of the
                                ///< boundary value problem
    std::size_t dimension;      ///< Dimension of the boundary value problem
    mesh grid;                  ///< Mesh of the boundary value problem
};

template <typename F, typename Bc, typename Basis>
boundary_value_problem(F &&, Bc &&, Basis &&,
                       nld::collocations::mesh_parameters, std::size_t)
    -> boundary_value_problem<F, Bc, Basis>;

} // namespace nld::collocations
