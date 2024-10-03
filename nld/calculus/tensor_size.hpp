#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/calculus/fwd.hpp>

namespace nld {
/// @brief The meta function to determinate size of tensor.
/// @tparam T type of expression.
template <typename T>
struct tensor_size final {
    constexpr static std::size_t size = T::tensor_size;
};

/// @brief The meta function to determinate size of tensor for double scalar.
template <>
struct tensor_size<double> final {
    constexpr static std::size_t size = 0;
};

} // namespace nld
