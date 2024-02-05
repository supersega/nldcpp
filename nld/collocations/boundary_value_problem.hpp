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

namespace concepts {
template <typename Fn>
concept CollocationFunction = requires(Fn f) {
    {
        f(std::declval<nld::archetypes::vector>(),
          std::declval<nld::archetypes::scalar>(),
          std::declval<nld::archetypes::vector>())
    } -> nld::Vector;
};
} // namespace concepts

/// @brief Boundary value problem
/// @tparam F Type of the function
/// @tparam Bc Type of the boundary conditions
/// @tparam Basis Type of the basis builder
/// @details This class represents a boundary value problem of the form
/// \f[
///    \begin{cases}
///    \mathbf{u}'(t) = \mathbf{f}(\mathbf{u}(t), t, p) \\
///    \mathbf{b}(\mathbf{u}(t_0), \mathbf{u}(t_N)) = 0 \\
///    \end{cases}
/// \f]
template <concepts::CollocationFunction F, typename Bc, typename Basis>
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
          grid_(nld::collocations::mesh(
              parameters, nld::collocations::uniform_mesh_nodes,
              nld::collocations::legandre_collocation_points)) {}

    /// @frief just for testing
    template <typename Wrt, typename At, typename V>
    auto jacobian(Wrt &&wrt, At &&at, V &v) const -> nld::sparse_matrix_xd {
        auto u0 = std::get<0>(at);
        auto J_full = jacobian(u0, v);
        nld::sparse_matrix_xd J =
            J_full.block(0, 0, J_full.rows() - 1, J_full.cols() - 1);

        return J;
    }

    /// @brief Evaluate the jacobian of the boundary value problem
    /// @details The jacobian of the boundary value problem is the sparse
    /// * * * * 0 0 0 0 *
    /// * * * * 0 0 0 0 *
    /// * * * * * * * * 0
    /// 0 0 0 0 * * * * *
    /// 0 0 0 0 * * * * *
    /// * 0 0 0 0 0 0 * 0
    /// @tparam At Type of the variables to evaluate at
    /// @tparam V Type of the result of the function
    /// @param at Variables to evaluate at
    /// @param v Value of the function
    template <typename At, typename V>
    auto jacobian(At &u, V &v) const -> nld::sparse_matrix_xd {
        collocation_on_elements evaluator(grid_, basis_builder);

        auto N = parameters.intervals;
        auto m = parameters.collocation_points;
        std::size_t n = dimension;

        auto size = N * n * m + (N - 1) * n + n;
        auto parameters_size = u.size() - size;

        // Create square sparce Jacobian to avoid reallocation
        // on client side
        nld::sparse_matrix_xd result(size + parameters_size,
                                     size + parameters_size);
        Eigen::VectorXi size_estimation(size + parameters_size);
        // TODO: we know exact how many non-zero elements we have
        // so we can avoid reallocation/overestimation
        size_estimation << VectorXi::Constant(size, 3 * m * n),
            VectorXi::Constant(parameters_size, size);
        result.reserve(size_estimation);

        nld::vector_xdd f(size);
        f = nld::vector_xdd::Zero(size);

        auto parameters = u.tail(parameters_size);

        for (std::size_t j = 0; j < N; ++j) {
            auto values = evaluator.values_in_collocation_points(j);
            auto derivatives = evaluator.derivatives_in_collocation_points(j);

            // unknowns on element j
            auto u_j = u.segment(j * (m + 1) * n, (m + 1) * n);
            for (std::size_t k = 0; k < m; ++k) {
                auto t = grid_.collocation_points[j * m + k];

                // Next collocation equations: p'(t) - f(p(t), t, p) = 0
                // We sutisfy this equation on each finite element at
                // m collocation points
                auto collocation_equations =
                    [this, &values, &derivatives, n, m,
                     k](const auto &u_j, auto t,
                        const auto &parameters) -> nld::vector_xdd {
                    // Build polynomial and its derivative
                    nld::vector_xdd p = nld::vector_xdd::Zero(n);
                    nld::vector_xdd p_prime = nld::vector_xdd::Zero(n);
                    for (std::size_t l = 0; l < m + 1; ++l) {
                        for (std::size_t i = 0; i < n; ++i) {
                            auto q = u_j[l * n + i];
                            p[i] += values(l, k) * q;
                            p_prime[i] += derivatives(l, k) * q;
                        }
                    }

                    auto f_p = function(p, t, parameters);
                    return p_prime - f_p;
                };

                nld::vector_xdd value;
                auto J = autodiff::forward::jacobian(
                    collocation_equations, nld::wrt(u_j, parameters),
                    nld::at(u_j, t, parameters), value);

                // Fill Jacobian w.r.t. u_j
                auto shift = j * (m + 1) * n + k * n;
                for (std::size_t l = 0; l < m + 1; ++l) {
                    for (std::size_t c = 0; c < n; ++c) {
                        for (std::size_t i = 0; i < n; ++i) {
                            auto rl = i;
                            auto cl = l * n + c;
                            auto rg = j * (m + 1) * n + k * n + rl;
                            auto cg = j * (m + 1) * n + cl;
                            result.insert(rg, cg) = J(rl, cl);
                        }
                    }
                }

                // Fill Jacobian w.r.t. parameters
                for (std::size_t c = 0; c < parameters_size; ++c) {
                    for (std::size_t i = 0; i < n; ++i) {
                        auto rl = i;
                        auto cl = (m + 1) * n + c;
                        auto rg = j * (m + 1) * n + k * n + rl;
                        auto cg = size + c;
                        result.insert(rg, cg) = J(rl, cl);
                    }
                }

                // Values of colloaction equations
                f.segment(j * (m + 1) * n + k * n, n) = value;
            }

            if (j < N - 1) {
                auto values_in_mesh_node = evaluator.values_in_mesh_node(j + 1);

                // Continuity equations: Pm[j - 1](u_j) - Pm[j](u_j) = 0
                // On each finite element we have polynomial of degree m
                // Solution should be continuous on the mesh nodes, so we
                // add continuity equations to satisfy this condition
                auto continuity_equations =
                    [this, &values_in_mesh_node, n, m,
                     j](const auto &u_jl, const auto &u_jr) -> nld::vector_xdd {
                    nld::vector_xdd continuity = nld::vector_xdd::Zero(n);
                    for (std::size_t l = 0; l < m + 1; ++l) {
                        for (std::size_t i = 0; i < n; ++i) {
                            auto ql = u_jl[l * n + i];
                            auto qr = u_jr[l * n + i];
                            continuity[i] += values_in_mesh_node(l, 0) * ql -
                                             values_in_mesh_node(l, 1) * qr;
                        }
                    }
                    return continuity;
                };

                auto u_jl = u.segment(j * (m + 1) * n, (m + 1) * n);
                auto u_jr = u.segment((j + 1) * (m + 1) * n, (m + 1) * n);

                nld::vector_xdd value;
                auto J = autodiff::forward::jacobian(
                    continuity_equations, nld::wrt(u_jl, u_jr),
                    nld::at(u_jl, u_jr), value);

                // Fill Jacobian w.r.t. u_jl for continuity equations
                auto shift = j * (m + 1) * n + m * n;
                for (std::size_t l = 0; l < 2 * (m + 1); ++l) {
                    for (std::size_t c = 0; c < n; ++c) {
                        for (std::size_t i = 0; i < n; ++i) {
                            auto rl = i;
                            auto cl = l * n + c;
                            auto rg = j * (m + 1) * n + m * n + rl;
                            auto cg = j * (m + 1) * n + cl;
                            result.insert(rg, cg) = J(rl, cl);
                        }
                    }
                }

                // Fill values of continuity equations, since we have
                // continuous solution on the mesh nodes, values should be 0
                // but we write them strictly now
                f.segment(j * (m + 1) * n + m * n, n) = value;
            }
        }

        // Boundary conditions equations: b(u0, uN) = 0
        // For periodic boundary conditions we have b(u0, uN) = u0 - uN = 0
        // this leads to two identity matrices in Jacobian, so point for
        // optimization
        auto boundary_conditions_equations =
            [this, n](const auto &u0, const auto &uN) -> nld::vector_xdd {
            return boundary_conditions(u0, uN);
        };

        auto u0 = u.head(n);
        auto uN = u.segment(N * m * n + (N - 1) * n, n);
        nld::vector_xdd value;
        auto J = autodiff::forward::jacobian(boundary_conditions_equations,
                                             nld::wrt(u0, uN), nld::at(u0, uN),
                                             value);

        // Fill Jacobian w.r.t. u0, uN for boundary conditions equations
        for (std::size_t l = 0; l < 2; ++l) {
            for (std::size_t c = 0; c < n; ++c) {
                for (std::size_t i = 0; i < n; ++i) {
                    auto rl = i;
                    auto cl = l * n + c;
                    auto rg = N * m * n + (N - 1) * n + rl;
                    auto cg = N * m * n * l + (N - 1) * n * l + c;
                    result.insert(rg, cg) = J(rl, cl);
                }
            }
        }

        // Fill values of boundary conditions equations
        // Again, should be zero, but we write them strictly now
        f.segment(N * m * n + (N - 1) * n, n) = value;

        v = f;
        return result;
    }

    /// @brief Evaluate the boundary value problem
    /// @tparam U Type of the variables to evaluate at
    /// @param u Variables to evaluate at
    /// @return Value of the boundary value problem
    auto operator()(const nld::vector_xdd &u) const -> nld::vector_xdd {
        nld::collocations::collocation_on_elements evaluator(grid_,
                                                             basis_builder);

        auto N = parameters.intervals;
        auto m = parameters.collocation_points;
        std::size_t n = dimension;

        nld::vector_xdd f(N * n * m + (N - 1) * n + n);
        f = nld::vector_xdd::Zero(N * n * m + (N - 1) * n + n);

        // TODO: add support for multiple parameters
        auto parameters = u.tail(1);

        for (std::size_t j = 0; j < N; ++j) {
            auto values = evaluator.values_in_collocation_points(j);
            auto derivatives = evaluator.derivatives_in_collocation_points(j);

            for (std::size_t k = 0; k < m; ++k) {
                auto t = grid_.collocation_points[j * m + k];

                nld::vector_xdd p = nld::vector_xdd::Zero(n);
                nld::vector_xdd p_prime = nld::vector_xdd::Zero(n);
                for (std::size_t l = 0; l < m + 1; ++l) {
                    for (std::size_t i = 0; i < n; ++i) {
                        auto q = u[j * (m + 1) * n + l * n + i];
                        p[i] += values(l, k) * q;
                        p_prime[i] += derivatives(l, k) * q;
                    }
                }

                auto f_p = function(p, t, parameters);
                f.segment(j * (m + 1) * n + k * n, n) = p_prime - f_p;
            }

            if (j < N - 1) {
                auto values_in_mesh_node = evaluator.values_in_mesh_node(j + 1);

                nld::vector_xdd continuity = nld::vector_xdd::Zero(n);
                for (std::size_t l = 0; l < m + 1; ++l) {
                    for (std::size_t i = 0; i < n; ++i) {
                        auto ql = u[j * (m + 1) * n + l * n + i];
                        auto qr = u[(j + 1) * (m + 1) * n + l * n + i];
                        continuity[i] += values_in_mesh_node(l, 0) * ql -
                                         values_in_mesh_node(l, 1) * qr;
                    }
                }

                f.segment(j * (m + 1) * n + m * n, n) = continuity;
            }
        }

        f.segment(N * m * n + (N - 1) * n, n) =
            boundary_conditions(u.head(n), u.tail(n + 1).head(n));

        return f;
    }

    /// @brief mesh of the boundary value boundary_value_problem
    /// @return mesh of the boundary value boundary_value_problem
    [[nodiscard]] constexpr const auto &grid() const noexcept { return grid_; }

    /// @brief the underlying function of the boundary value
    /// boundary_value_problem
    /// @return underlying function
    [[nodiscard]] auto underlying_function() const noexcept -> const F & {
        return function;
    }

private:
    F function;                 ///< Function of the boundary value problem
    Bc boundary_conditions;     ///< Boundary conditions of the boundary value
                                ///< problem
    Basis basis_builder;        ///< Basis builder of the boundary value problem
    mesh_parameters parameters; ///< Mesh parameters of the
                                ///< boundary value problem
    std::size_t dimension;      ///< Dimension of the boundary value problem
    mesh grid_;                 ///< Mesh of the boundary value problem
};

template <concepts::CollocationFunction F, typename Bc, typename Basis>
boundary_value_problem(F &&, Bc &&, Basis &&,
                       nld::collocations::mesh_parameters, std::size_t)
    -> boundary_value_problem<F, Bc, Basis>;

} // namespace nld::collocations
