#pragma once

#include <utility>

#include <nld/calculus/concepts.hpp>
#include <nld/math.hpp>

namespace nld {
template <typename F>
struct scalar_function final {
    /// @brief Tensor size of test functions.
    static constexpr nld::index tensor_size = 0;

    /// @brief Construct a new scalar function object
    /// @param function
    explicit scalar_function(F &&function)
        requires(ScalarFunction<F>)
        : function(std::forward<F>(function)) {}

    /// @brief Compute scalar function.
    /// @param i unused.
    /// @return Function f(x) -> adnum.
    auto value() const { return function; }

    /// @brief The value scalar function.
    /// @param i unused.
    /// @return Function f(x) -> adnum to compute function at point.
    auto value([[maybe_unused]] std::tuple<> i) const { return value(); }

    /// @brief This is a scalar function, it does not depended on test
    /// functions.
    auto count() const { return std::tuple<>(); }

private:
    F function;
};

template <typename F>
scalar_function(F &&) -> scalar_function<F>;

} // namespace nld
