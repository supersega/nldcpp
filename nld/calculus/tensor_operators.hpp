#pragma once

#include <nld/calculus/concepts.hpp>
#include <nld/calculus/delta_function.hpp>
#include <nld/calculus/fwd.hpp>
#include <nld/calculus/tensor_expression.hpp>

namespace nld {
/// @brief Multiply tensor expressions.
/// @tparam L Type of the left expression.
/// @tparam R Type of the right expression.
/// @param l left expression.
/// @param r right expression.
/// @return mul type holds l and r.
template <typename L, typename R>
    requires(TensorMulOperands<std::remove_reference_t<L>,
                               std::remove_reference_t<R>>)
constexpr auto operator*(L &&l, R &&r) -> nld::tensor_mul<L, R> {
    return {std::forward<L>(l), std::forward<R>(r)};
}

/// @brief Multiply tensor expressions.
/// @tparam L Type of the left expression.
/// @tparam R Type of the right expression.
/// @param l left expression.
/// @param r right expression.
/// @return mul type holds l and r.
template <typename E, typename D, typename R>
    requires(TensorExpression<std::remove_reference_t<R>>)
constexpr auto operator*(nld::integral<E, D> &&l, R &&r)
    -> nld::mul<nld::integral<E, D>, R> {
    return {std::forward<nld::integral<E, D>>(l), std::forward<R>(r)};
}

/// @brief Multiply tensor expressions.
/// @tparam L Type of the left expression.
/// @tparam R Type of the right expression.
/// @param l left expression.
/// @param r right expression.
/// @return mul type holds l and r.
template <typename L, typename R, typename T>
    requires(TensorExpression<std::remove_reference_t<T>>)
constexpr auto operator*(nld::mul<L, R> &&l, T &&r) -> nld::mul<
    std::remove_reference_t<decltype(l.left)>,
    decltype(std::forward<std::remove_reference_t<decltype(l.right)>>(l.right) *
             std::forward<T>(r))> {
    return {std::forward<std::remove_reference_t<decltype(l.left)>>(l.left),
            std::forward<std::remove_reference_t<decltype(l.right)>>(l.right) *
                std::forward<T>(r)};
}

/// @brief Multiply basis functions by bc.
/// @tparam B Boundary condition type.
/// @tparam Bases Basis type.
/// @param bc Boundary conditions.
/// @param tf the test functions.
/// @return weighted_test_functions what represents tests function with weihgt.
template <BoundaryCondition B, typename Space, typename Bases>
constexpr auto operator*(B bc, test_functions<Space, Bases> tf)
    -> weighted_test_functions<test_functions<Space, Bases>, B> {
    return weighted_test_functions{tf, bc};
}

/// @brief Multiply expression functions by dirac .
/// @param bc Boundary conditions.
/// @param tf the test functions.
/// @return weighted_test_functions what represents tests function with weihgt.
template <TensorExpression E, typename P>
constexpr auto operator*(E e, delta_function<P> df)
    -> dirac_shift<E, delta_function<P>> {
    return dirac_shift<E, delta_function<P>>{std::forward<E>(e), std::move(df)};
}
} // namespace nld
