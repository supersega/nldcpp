#pragma once

#include <nld/collocations/boundary_value_problem.hpp>
#include <nld/systems/periodic.hpp>
#include <nld/systems/periodic_base.hpp>

namespace nld {

namespace _cl = ::nld::collocations;

namespace internal {
template <typename Ds>
concept DoedelEquationNeeded = Ds::is_periodic_constraint_needed;
} // namespace internal

/// TODO: Add Doedel equation support
/// @brief Periodic collocations
/// @tparam F dynamical system
template <typename F, typename Basis>
struct periodic_collocations {
private:
    struct bc {
        auto operator()(const auto &u0, const auto &uN) const {
            return u0 - uN;
        }
    };

public:
    /// @brief Construct a new periodic collocations object
    /// @param f dynamical system
    /// @param parameters mesh parameters
    /// @param dimension dimension of the dynamical system
    explicit periodic_collocations(F &&f, Basis &&basis,
                                   nld::_cl::mesh_parameters parameters,
                                   std::size_t dimension)
        : parameters(parameters), dimension_(dimension),
          bvp(std::forward<F>(f), bc{}, std::forward<Basis>(basis), parameters,
              dimension) {}

    /// @brief Evaluate the Jacobian matrix of the dynamical system
    /// @tparam At type of the argument of the dynamical system
    /// @tparam V type of the vector where we evaluate the Jacobian matrix
    /// @param at argument of the dynamical system
    /// @param v vector where we evaluate the Jacobian matrix
    /// @return Jacobian matrix of the dynamical system
    template <typename At, typename V>
    auto jacobian(At &at, V &v) const
        requires(!internal::DoedelEquationNeeded<F>)
    {
        return bvp.jacobian(at, v);
    }

    /// @brief Evaluate the dynamical system
    /// @tparam Vector vector type
    /// @param u vector where we evaluate the dynamical system
    /// @return value of the dynamical system
    template <typename Vector>
    auto operator()(const Vector &u) const -> Vector
        requires(!internal::DoedelEquationNeeded<F>)
    {
        return bvp(u);
    }

    /// @brief Get dimension of the dynamical systems
    /// @return dimension of the dynamical systems
    [[nodiscard]] constexpr auto dimension() const noexcept {
        return dimension_;
    }

    /// @brief Get mesh parameters
    /// @return mesh parameters
    [[nodiscard]] constexpr const auto &mesh_parameters() const noexcept {
        return parameters;
    }

    /// @brief Get mesh
    /// @return mesh
    [[nodiscard]] constexpr const auto &grid() const noexcept {
        return bvp.grid();
    }

    /// @brief Get underlying function
    /// @return underlying function
    [[nodiscard]] constexpr const auto &underlying_function() const noexcept {
        return bvp.underlying_function();
    }

private:
    using bvp_t = nld::_cl::boundary_value_problem<F, bc, Basis>;
    nld::_cl::mesh_parameters parameters; ///< Mesh parameters
    std::size_t dimension_;               ///< Dimension of the dynamical system
    bvp_t bvp; ///< Boundary value problem with abstract boundary conditions
};

template <typename F, typename Basis>
periodic_collocations(F &&f, Basis &&basis,
                      nld::collocations::mesh_parameters parameters,
                      std::size_t dimension) -> periodic_collocations<F, Basis>;

/// @brief is system desctitized by collocaiton method.
template <typename F, typename Basis>
struct is_collocation_discretization<periodic_collocations<F, Basis>>
    : std::true_type {};

} // namespace nld
