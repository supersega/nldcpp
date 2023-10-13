#pragma once 

#include <nld/autocont/jacobian_mixin.hpp>

namespace nld {

/// TODO: Consider the way to propagate previous solution into function.

/// @brief Convex homotopy function.
/// @details This function will be used for identifying of initial guess 
/// for nonlinear problems, and for disconnected branches detections.
template<typename F, typename H>
struct convex_homotopy final : jacobian_mixin<convex_homotopy<F, H>> {
    /// @brief Create convex homotopy function.
    /// @param f nonlinear function.
    /// @param h homotopy function.
    explicit convex_homotopy(F&& f, H&& h) : function{ std::forward<F>(f) }, homotopy{ std::forward<H>(h) } { }

    /// @brief Convex homotopy function call operator.
    /// @param variables - N+1 dimensional point where we want to evaluate function.
    /// @return Value of homotopy function at given point.
    template<typename Vec>
    auto operator() (const Vec& variables) const -> Vec {
        const auto dim = variables.size();
        auto kappa = variables(dim - 1);
        Vec unknowns = variables.head(dim - 1);

        return kappa * function(unknowns) - (1.0 - kappa) * homotopy(unknowns);
    }
private:
    F function; ///< nonlinear function.
    H homotopy; ///< homotopy function.
};

template<typename D, typename V>
convex_homotopy(D&&, V&&)->convex_homotopy<D, V>;

/// @brief Convex homotopy function.
/// @details This function will be used for identifying of initial guess 
/// for nonlinear problems, and for disconnected branches detections.
template<typename F, typename D>
struct newton_homotopy final : jacobian_mixin<newton_homotopy<F, D>>{
    /// @brief Create convex homotopy function.
    /// @param f nonlinear function.
    /// @param h homotopy function.
    explicit newton_homotopy(F&& f, D&& d) : function{ std::forward<F>(f) }, d_trick{ std::forward<D>(d) } { }

    /// @brief Convex homotopy function call operator.
    /// @param variables - N+1 dimensional point where we want to evaluate function.
    /// @return Value of homotopy function at given point.
    template<typename Vec>
    auto operator() (const Vec& variables) const -> Vec {
        const auto dim = variables.size();
        auto kappa = variables(dim - 1);
        Vec unknowns = variables.head(dim - 1);

        return function(unknowns) - (1.0 - kappa) * function(d_trick);
    }
private:
    F function; ///< nonlinear function.
    D d_trick;  ///< homotopy function.
};

template<typename D, typename V>
newton_homotopy(D&&, V&&)->newton_homotopy<D, V>;
}