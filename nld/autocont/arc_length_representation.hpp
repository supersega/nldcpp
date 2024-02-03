#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <iostream>

namespace nld {
namespace detail {

inline auto build_matrix(nld::matrix_xd &J, const nld::vector_xd &v) -> void {
    auto dim = v.size();
    J.conservativeResize(dim, dim);
    J.bottomRows(1) = v.transpose();
}

inline auto build_matrix(nld::sparse_matrix_xd &J, const nld::vector_xd &v)
    -> void {
    auto dim = v.size();

    J.conservativeResize(dim, dim);
    for (int i = 0; i < v.size(); ++i)
        J.insert(dim - 1, i) = v(i);
}
} // namespace detail

/// @brief Class which represent function as nonlinear function with arc length
/// parametrization.
/// @details This class can be used for pseudo arc length continuation
/// algorithm. We assume that functions agreed with library interfaces. The
/// function should have the next signature: fn: Vector(N+1) -> Vector(N), where
/// the last input argument is a continuation parameter.
/// @see concepts.hpp
/// @tparam F function.
/// @tparam V N dimensional vector.
/// @tparam Real real number.
template <typename F, typename V, typename Real>
struct arc_length_representation final {
    /// @brief Create arc length parametrization of function.
    /// @param function nonlinear function.
    /// @param point computed point on previous continuation step.
    /// @param tangent computed tangent on previous continuation step.
    /// @param ds step to compute arc length equation.
    explicit arc_length_representation(F &&fn, const V &point, const V &tangent,
                                       Real ds)
        : function{std::forward<F>(fn)}, previous_point{point},
          previous_tangent{tangent}, step{ds} {
        if constexpr (nld::FunctionWithPreviousStepSolution<F>)
            function.set_previous_solution(previous_point);
    }

    /// @brief Get underlying function.
    /// @details Get nonlinear function with next signature: fn(Vector) ->
    /// Vector, Useful in map function.s
    /// @return underlying nonlinear function.
    auto underlying_function() -> F & { return function; }

    /// @brief Nonlinear system plus Keller's arc length equation.
    /// @details Evaluate nonlinear function at given ponit,
    /// then compute Keller's equation in that point.
    /// @param variables - N+1 dimensional point where we want to evaluate
    /// function.
    /// @return Value of nonlinear function at given point.
    template <typename Vec>
    auto operator()(Vec &variables) const {
        const auto dim = variables.size();

        Vec value(dim);
        value.head(dim - 1) = function(variables);
        value(dim - 1) = arc_length_equation(variables);

        return value;
    }

    /// @brief Jacobian of dynamic system.
    /// @details Jacobian function is implemented to allows us
    /// using custom Jacobian, by implementing it in 'function'.
    /// Custom nonlinear functions can use jacobian_mixin class
    /// to inject required jacobian function.
    /// @param at Point where we evaluate jacobian.
    /// @param v Value of function if given point.
    /// @return Jacobi matrix.
    template <typename At, typename Result>
    auto jacobian(At &at, Result &v) const {
        using Matrix = decltype(function.jacobian(at, v));

        const auto dim = at.size();
        v.resize(dim);

        auto value = v.head(dim - 1);

        auto keller = [this](auto &val) { return arc_length_equation(val); };
        auto dkeller = autodiff::forward::gradient(keller, nld::wrt(at),
                                                   nld::at(at), v(dim - 1));

        auto top_left = function.jacobian(at, value);
        detail::build_matrix(top_left, dkeller);

        return top_left;
    }

    /// @brief Tangential to bifurcation curve at given point.
    /// @param variables N+1 unknown variables.
    /// @return Tangential at point variables.
    template <typename Vec>
    auto tangential(Vec &variables) const -> Vec {
        // Tangential can be evaluated using Jacobi matrix of AL representation
        using Result = decltype(std::apply(*this, at(variables)));

        Result result;
        auto jac = this->jacobian(variables, result);

        const auto dim = jac.rows();

        nld::vector_xd right = nld::vector_xd::Zero(dim);
        right(dim - 1) = 1.0;

        // Solve system to compute tangent
        // (df/dy df/dlambda)   = (0)
        // (dy0/ds dlambda0/ds) = (1)
        Vec tan = nld::math::linear_algebra::solve(jac, right);
        tan.normalize();

        return tan;
    }

    /// @brief Norm of function at given point.
    /// @param at point.
    /// @return Norm at point.
    template <typename At>
    auto norm(At &&at) const {
        auto value = std::apply(*this, at);
        return value.head(value.size() - 1).norm();
    }

    /// @brief Set previous solution.
    /// @details This will be called on Newton iteration. We got to eval some
    /// equations, which e.g. Period condition on each iteration. So, all
    /// supported functions must implement this method, and we will just
    /// propagate this to target.
    /// @param solution - previous solution
    template <typename Vec, typename Fo = F,
              std::enable_if_t<FunctionWithPreviousStepSolution<Fo>>...>
    void set_previous_solution(const Vec &solution) {
        function.set_previous_solution(solution);
    }

    /// @brief Set previous computed point and tangent on branch.
    /// @details This will be called on continuation iteration.
    /// All data which we need to build Keller's arc length data.
    /// @param point - previous point on branch.
    /// @param tangent - previous tangent at point on branch.
    template <typename Vec>
    void set_arc_length_properties(const Vec &point, const Vec &tangent,
                                   Real ds) {
        previous_point = point;
        previous_tangent = tangent;
        step = ds;
    }

private:
    template <typename Vec>
    auto arc_length_equation(Vec &variables) const {
        auto keller = (variables - previous_point).dot(previous_tangent) - step;
        return keller;
    }

    F function;         ///< Dynamic system wrapper.
    V previous_point;   ///< Previous point on branch (u).
    V previous_tangent; ///< Tangent at previous point on branch (du/ds).
    Real step;          ///< Arc length step (delta(s)).
};

template <typename D, typename V, typename F>
arc_length_representation(D &&, const V &, const V &, F)
    -> arc_length_representation<D, V, F>;
} // namespace nld
