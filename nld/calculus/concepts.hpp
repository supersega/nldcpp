#pragma once

#include <nld/calculus/adnum.hpp>
#include <nld/calculus/fwd.hpp>

namespace nld {
namespace archetypes {
/// @brief Archetype describes scalar function what we can integrate.
struct scalar_function final {
    template <typename Scalar>
    auto operator()(Scalar) -> Scalar;

    template <typename Scalar>
    auto operator()(Scalar, Scalar) -> Scalar;

    template <typename Scalar>
    auto operator()(Scalar, Scalar, Scalar) -> Scalar;
};

/// @brief Archetype describes domain of the integration.
struct domain final {};
} // namespace archetypes

/// @brief One dimension scalar function supported for calculus.
/// @tparam F callable function
template <typename F>
concept OneDimensionScalarFunction = requires(F f) {
    { f(std::declval<adnum>()) } -> std::same_as<adnum>;
};

/// @brief Two dimension scalar function supported for calculus.
/// @tparam F callable function.
template <typename F>
concept TwoDimensionScalarFunction = requires(F f) {
    { f(std::declval<adnum>(), std::declval<adnum>()) } -> std::same_as<adnum>;
};

/// @brief Three dimension scalar function supported for calculus.
/// @tparam F callable function.
template <typename F>
concept ThreeDimensionScalarFunction = requires(F f) {
    {
        f(std::declval<adnum>(), std::declval<adnum>(), std::declval<adnum>())
    } -> std::same_as<adnum>;
};

/// @brief Three scalar function supported for calculus.
/// @tparam F callable function.
template <typename F>
concept ScalarFunction =
    OneDimensionScalarFunction<F> || TwoDimensionScalarFunction<F> ||
    ThreeDimensionScalarFunction<F>;

namespace internal {
template <typename T>
struct is_expression : std::false_type {};

template <typename Space, typename Basis>
struct is_expression<test_functions<Space, Basis>> : std::true_type {};

template <typename T, typename B>
struct is_expression<weighted_test_functions<T, B>> : std::true_type {};

template <typename L, typename R>
struct is_expression<add<L, R>> : std::true_type {};

template <typename L, typename R>
struct is_expression<tensor_mul<L, R>> : std::true_type {};

template <typename E, typename W>
struct is_expression<derivative<E, W>> : std::true_type {};

template <typename F>
struct is_expression<scalar_function<F>> : std::true_type {};

template <typename F>
struct is_expression<eigenfunctions<F>> : std::true_type {};

template <typename E, typename I, FunctionalDomain1d D, typename O>
struct is_expression<variable_integral<E, I, D, O>> : std::true_type {};

template <typename T>
struct is_integral : std::false_type {};

template <typename E, typename D>
struct is_integral<integral<E, D>> : std::true_type {};

template <typename T>
struct is_scalar_function_expression : std::false_type {};

template <typename F>
struct is_scalar_function_expression<scalar_function<F>> : std::true_type {};

template <typename T>
struct are_all_with_subdomains : std::false_type {};

/// @brief Basis functions has their own domains.
/// @tparam T type.
template <typename T>
concept BasisWithFunctionsOnSubdomains =
    std::is_base_of_v<basis, T> && requires(T b) {
        { b.subdomain(std::declval<nld::index>()) };
    };

template <BasisWithFunctionsOnSubdomains... Args>
struct are_all_with_subdomains<std::tuple<Args...>> : std::true_type {};

template <typename T>
struct is_subdomain_defined : std::false_type {};

template <typename Space, typename Basis>
struct is_subdomain_defined<test_functions<Space, Basis>>
    : std::conditional_t<test_functions<Space, Basis>::with_subdomains,
                         std::true_type, std::false_type> {};

template <typename T, typename B>
struct is_subdomain_defined<weighted_test_functions<T, B>>
    : std::conditional_t<std::remove_reference_t<T>::with_subdomains,
                         std::true_type, std::false_type> {};

template <typename L, typename R>
struct is_subdomain_defined<tensor_mul<L, R>>
    : std::conditional_t<
          is_subdomain_defined<std::remove_reference_t<R>>::value,
          std::true_type, std::false_type> {};

template <typename L, typename F>
struct is_subdomain_defined<tensor_mul<L, nld::scalar_function<F>>>
    : std::conditional_t<std::is_arithmetic<std::remove_reference_t<L>>::value,
                         std::true_type, std::false_type> {};

template <typename E, typename W>
struct is_subdomain_defined<derivative<E, W>>
    : std::conditional_t<
          is_subdomain_defined<std::remove_reference_t<E>>::value,
          std::true_type, std::false_type> {};

template <typename F>
struct is_subdomain_defined<scalar_function<F>> : std::false_type {};

template <typename F>
struct is_subdomain_defined<eigenfunctions<F>> : std::false_type {};
} // namespace internal

/// @brief TensorExpression concept to define expressions.
/// @tparam T type.
template <typename T>
concept TensorExpression = nld::internal::is_expression<T>::value;

/// @brief Constant in terms of tensors.
/// @tparam T type.
template <typename T>
concept Constant = std::is_arithmetic<T>::value;

/// @brief Integral.
/// @tparam T type.
template <typename T>
concept Integral = nld::internal::is_integral<T>::value;

/// @brief Basis functions has their own domains.
/// @tparam T type.
template <typename T>
concept AllWithSubdomains = nld::internal::are_all_with_subdomains<T>::value;

/// @brief Integral.
/// @tparam T type.
template <typename T>
concept SubdomainDefined = nld::internal::is_subdomain_defined<T>::value;

/// @brief Mul expression operands.
/// @tparam T type.
template <typename L, typename R>
concept TensorMulOperands = (TensorExpression<L> && TensorExpression<R>) ||
                            (Constant<L> && TensorExpression<R>) ||
                            (TensorExpression<L> && Constant<R>);

/// @brief TensorExpression concept to define expressions.
/// @tparam T type.
template <typename T>
concept ScalarFunctionExpression =
    nld::internal::is_scalar_function_expression<T>::value;

/// @brief Boundary condition.
/// @tparam T type.
template <typename T>
concept BoundaryCondition = std::is_base_of_v<boundary_condition, T>;

/// @brief Basis functions concept.
/// @tparam T type of basis.
template <typename T>
concept BasisFunctions = std::is_base_of_v<basis, T> && requires(T bf) {
    { bf.template derivative<0>(std::declval<nld::index>()) };
    { bf.template derivative<1>(std::declval<nld::index>()) };
    { bf.template derivative<2>(std::declval<nld::index>()) };
};

/// @brief Integration concept.
/// @details Integration should provide static function to evaluate integral.
/// See arguments bellow to understand signature.
template <typename S>
concept Integrator = requires(S s) {
    typename S::integration_parameters_t;
    {
        S::integrate(std::declval<archetypes::scalar_function>(),
                     std::declval<typename S::integration_parameters_t>(),
                     std::declval<archetypes::domain>())
    } -> Scalar;
};
} // namespace nld
