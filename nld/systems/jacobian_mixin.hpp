#pragma once

#include <functional>
#include <type_traits>

#include <Eigen/Dense>

#include <nld/core/aliases.hpp>
#include <nld/core/utils.hpp>

namespace nld {

/// TODO: Consider allocators for Eigen vectors and matrices.
/// Now we have a lot memory allocation on heap.

/// @brief Class to inject jacobian functions for nonlinear functions using
/// operator().
/// @details We use boundary_value_problem in this class, and it can use any
/// @tparam F Function.
/// @tparam P Evaluation parameters.
template <typename F>
struct jacobian_mixin : nld::utils::crtp<F> {
    /// @brief Jacobian of dynamic system.
    /// @param wrt Variables w.r.t. we evaluate jacobian.
    /// @param at Point where we evaluate jacobian.
    /// @param v Value of function if given point.
    /// @return Jacobi matrix.
    template <typename Wrt, typename At, typename Result>
    auto jacobian(Wrt &&wrt, At &&at, Result &v) const {
        return autodiff::jacobian(this->derived(), wrt, at, v);
    }

    /// @brief Jacobian of dynamic system.
    /// @details This function is used by continuation methods.
    /// This function give a guarantee that at and wrt are the same. So we can
    /// perform some optimizations.
    /// @param at Point where we evaluate jacobian.
    /// @param v Value of function if given point.
    /// @return Jacobi matrix.
    template <typename At, typename Result>
    auto jacobian(At &at, Result &v) const {
        return autodiff::jacobian(this->derived(), nld::wrt(at), nld::at(at),
                                  v);
    }
};
} // namespace nld
