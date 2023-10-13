#pragma once 

namespace nld {
/// @brief The meta function to determinate size of tensor.
/// @tparam T type of expression.
template<typename T>
struct tensor_size;

/// @brief The meta function to determinate size of tensor for double scalar.
template<>
struct tensor_size<double> final {
    constexpr static std::size_t size = 0;
};

/// @brief The meta function to determinate size of mul expression tensor.
/// @details It is e product of underlying dimenstions.
template<typename L, typename R>
struct tensor_size<tensor_mul<L, R>> final {
    constexpr static std::size_t size = tensor_size<std::remove_reference_t<L>>::size + tensor_size<std::remove_reference_t<R>>::size;
};

/// @brief The meta function to determinate size of test function expression tensor.
/// @details Test functions are one dim tensor.
template<typename Space, typename Basis>
struct tensor_size<test_functions<Space, Basis>> final {
    constexpr static std::size_t size = 1;
};

/// @brief The meta function to determinate size of weighted test function expression tensor.
/// @details Weighted test functions are one dim tensor.
template<typename T, typename B>
struct tensor_size<weighted_test_functions<T, B>> final {
    constexpr static std::size_t size = 1;
};

/// @brief The meta function to determinate size of derivative expression tensor.
/// @details It is e product of underlying dimenstions.
template<typename E, typename W>
struct tensor_size<derivative<E, W>> final {
    constexpr static std::size_t size = tensor_size<std::remove_reference_t<E>>::size;
};

/// @brief The meta function to determinate size of test function expression tensor.
/// @details Test functions are one dim tensor.
template<typename T>
struct tensor_size<eigenfunctions<T>> final {
    constexpr static std::size_t size = 1;
};

/// @brief The meta function to determinate size of tensor for scalar function.
/// @details Scalar function it is just a scalar but callable.
template<typename F>
struct tensor_size<scalar_function<F>> final {
    constexpr static std::size_t size = 0;
};

/// @brief The meta function to determinate size of derivative expression tensor.
/// @details It is e product of underlying dimenstions.
template<typename E, typename F>
struct tensor_size<dirac_shift<E, F>> final {
    constexpr static std::size_t size = tensor_size<std::remove_reference_t<E>>::size;
};
}
