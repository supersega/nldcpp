#pragma once

#include <nld/core.hpp>

#include <iostream>
#include <nld/math/linear_algebra.hpp>
#include <nld/math/newton_parameters.hpp>

namespace nld::math {

namespace detail {
/// Forward declaration of Newton step. Internal function, should
/// not be used in user code. In C++ it would not be exported.
template <typename F, typename Wrt, typename At, typename Float>
bool newton_step(F &&f, Wrt &&initial, At &&at, Float tolerance);

/// @brief Implementation of Newton method, which comute jacobian matrix
/// in all input points.
template <typename F, typename At, typename Float>
bool newton_step(F &&f, At &at, Float tolerance);

/// @brief Moore-Penrose-Newton correction implementation.
/// @details Moore-Penrose method a bit complicated, we don't able
/// to create representation class for it. So we create separate
/// function for Moore-Penrose correction.
template <typename F, typename Wrt, typename Tan, typename At, typename Float>
bool moore_penrose_newton_step(F &&f, Wrt &&initial_guess,
                               Tan &&tangential_guess, At &&at,
                               Float tolerance);
} // end namespace detail

/// Newton method function
/// @details
/// @param function Nonlinear function.
/// @param wrt Variables which will be variated.
/// @param at Point where function should be evaluated.
/// @param parameters Newton method parameters.
/// @return Result of computation.
template <typename F, typename Wrt, typename At, typename P>
[[nodiscard]] auto newton(F &&f, Wrt &&wrt, At &&at, P parameters)
    -> newton_info {
    // Loop of Newton method
    using std::forward;
    auto [max_iterations, tolerance] = parameters;
    while (max_iterations--)
        if (detail::newton_step(f, wrt, at, tolerance))
            return {parameters.max_iterations - max_iterations, true};
    // If iterations not success return false
    return {parameters.max_iterations, false};
}

/// Newton method function
/// @param function Nonlinear function.
/// @param at Point where function should be evaluated.
/// @param parameters Newton method parameters.
/// @return Result of computation.
template <typename F, typename At, typename P>
[[nodiscard]] auto newton(F &&f, At &at, P parameters) -> newton_info {
    // Loop of Newton method
    using std::forward;
    auto [max_iterations, tolerance] = parameters;
    while (max_iterations--)
        if (detail::newton_step(f, at, tolerance))
            return {parameters.max_iterations - max_iterations, true};
    // If iterations not success return false
    return {parameters.max_iterations, false};
}

/// Moore-Penrose-Newton correction.
/// @details
/// We are planing to use this function only in internal routines.
/// We leave this fn in nld::math namespace.
/// @param function Nonlinear function.
/// @param wrt Variables which will be variated.
/// @param at Point where function should be evaluated.
/// @param parameters Newton method parameters.
/// @return Result of computation.
template <typename F, typename Wrt, typename Tan, typename At, typename P>
[[nodiscard]] auto moore_penrose_newton(F &&f, Wrt &&wrt, Tan &&tan, At &&at,
                                        P parameters) -> newton_info {
    // Loop of Newton method
    auto [max_iterations, tolerance] = parameters;
    while (max_iterations--)
        if (detail::moore_penrose_newton_step(f, wrt, tan, at, tolerance))
            return {parameters.max_iterations - max_iterations, true};
    // If iterations not success return false
    return {parameters.max_iterations, false};
}

namespace detail {

// Compute norm of given function. If it scalar type just return sqrt.
template <typename F, typename At>
auto norm(const F &f, At &&at) {
    if constexpr (Norm<F, At>)
        return f.norm(at);
    else
        return std::apply(f, at.args).norm();
}

// Internal function to get jacobian. Need it since nld wrappers has jacobian
// method.
// TODO: check if it is called anywhere. If not remove it.
template <typename Function, typename Wrt, typename At, typename Result>
auto jacobian(const Function &f, Wrt &&wrt, At &&at, Result &&F) {
    if constexpr (Jacobian<Function, Wrt, At, Result>)
        return f.jacobian(wrt, at, F);
    else
        return autodiff::jacobian(f, wrt, at, F);
}

// Internal function to get jacobian. Need it since nld wrappers has jacobian
// method.
template <typename Function, typename At, typename Result>
auto jacobian(Function &f, At &at, Result &F) {
    if constexpr (JacobianFull<Function, At, Result>)
        return f.jacobian(at, F);
    else
        return autodiff::jacobian(f, nld::wrt(at), nld::at(at), F);
}

template <typename Function, typename At, typename Result>
auto gradient_finite_difference(Function &f, At &at, Result &F) {
    auto n = at.size();

    double h = 1e-3;

    nld::vector_xd grad(n);
    for (size_t i = 0; i < n; ++i) {
        nld::vector_xd xl = at;
        xl(i) -= h;
        nld::vector_xd xr = at;
        xr(i) += h;

        auto fl = std::invoke(f, xl);
        auto fr = std::invoke(f, xr);

        grad(i) = (fr - fl) / (2 * h);
        F = std::invoke(f, at);
    }

    return grad;
}

template <typename Function, typename At, typename Result>
auto gradient(Function &f, At &at, Result &F) {
    using number_t = decltype(std::invoke(f, at));
    if constexpr (std::is_floating_point_v<number_t>)
        return gradient_finite_difference(f, at, F);
    else
        return autodiff::gradient(f, nld::wrt(at), nld::at(at), F);
}

/// TODO: Avoid copy of vector (using xpr?) and optimize if `wrt` size == 1.
/// Make enable to use scalar functions.
template <typename F, typename Wrt, typename At, typename Float>
auto newton_step(F &&f, Wrt &&initial, At &&at, Float tolerance) -> bool {
    using Result = decltype(std::apply(f, at.args));

    Result value;
    auto jacobian = detail::jacobian(f, initial, at, value);

    Eigen::VectorXd result;
    nld::utils::tuple_to_vector(initial.args, result);
    Eigen::VectorXd previous_result = result;

    if constexpr (FunctionWithPreviousStepSolution<F>)
        f.set_previous_solution(result);

    // Make single step of Newton method

    result -= nld::math::linear_algebra::solve(jacobian,
                                               value.template cast<Float>());

    Eigen::VectorXd diff = result - previous_result;
    nld::utils::vector_to_tuple(result, initial.args);

    auto nf = norm(f, at);
    auto na = diff.norm() / result.norm();
    return nf < tolerance || na < tolerance;
}

template <typename F, typename At, typename Float>
auto newton_step(F &&f, At &at, Float tolerance) -> bool {
    using Result = decltype(std::invoke(f, at));

    Result value;
    auto jacobian = detail::jacobian(f, at, value);

    Eigen::VectorXd result = at.template cast<double>();
    Eigen::VectorXd previous_result = result;

    if constexpr (FunctionWithPreviousStepSolution<F>)
        f.set_previous_solution(result);

    // Make single step of Newton method

    result.head(value.size()) -= nld::math::linear_algebra::solve(
        jacobian, value.template cast<Float>());

    Eigen::VectorXd diff = result - previous_result;

    at = result;
    // std::cout << "Newton step: "
    //           << "111" << std::endl;
    auto nf = norm(f, nld::at(at));
    auto na = diff.norm() / result.norm();
    // std::cout << "Norm f: " << nf << " Norm a: " << na
    //           << " Norm arg: " << result.norm() << std::endl;
    return nf < tolerance || na < tolerance;
}

/// Implementation of Moore-Penrose-Newton step
template <typename F, typename Wrt, typename Tan, typename At, typename Float>
bool moore_penrose_newton_step(F &&f, Wrt &&wrt, Tan &&tan, At &&at,
                               Float tol) {
    auto wrts = count(wrt);
    nld::matrix_x<Float> jacobian(wrts, wrts);
    decltype(std::apply(f, at.args)) v(wrts);
    v(wrts - 1) = 0;
    auto Jxp = nld::math::detail::jacobian(f, wrt, at, v.head(wrts - 1));
    // Here we expect non square Jacobi matrix (wrts, wrts + 1)
    jacobian.topLeftCorner(wrts - 1, wrts) = Jxp;

    // Last row is a tangential, we use raw as a vector (Eigen is very cool)
    nld::utils::tuple_to_vector(tan, jacobian.row(wrts - 1));

    // Make single step of Newton method for "main variables"
    Eigen::VectorXd result;
    nld::utils::tuple_to_vector(wrt, result);

    result -= jacobian.fullPivHouseholderQr().solve(v.template cast<Float>());

    nld::utils::vector_to_tuple(result, wrt);

    // Make single step of Newton method for "tangential variables"
    nld::utils::tuple_to_vector(tan, result);

    v = Eigen::VectorXd::Zero(wrts);
    v(wrts - 1) = 1;

    result -= jacobian.fullPivHouseholderQr().solve(v.template cast<Float>());

    // This is our complicity... we should normalize solution after Newton step
    result.normalize();

    nld::utils::vector_to_tuple(result, tan);

    return norm(f, at) < tol;
}
} // end namespace detail
} // namespace nld::math

namespace nld {
using nld::math::newton;
}
