#pragma once

#include <concepts>

namespace nld {
/// @brief The 1-d domain with constant bounds.
template <typename T>
concept ConstantDomain1d = requires(T domain) {
    { domain.begin } -> std::convertible_to<double>;
    { domain.end } -> std::convertible_to<double>;
};

/// @brief The 1-d domain with variable bounds.
/// @details The domain with variable bounds might be used
/// only for inner integrals, while outer integral should
/// be calculated with constant bounds.
template <typename T>
concept FunctionalDomain1d = requires(T domain) {
    { domain.begin(std::declval<double>()) } -> std::same_as<double>;
    { domain.end(std::declval<double>()) } -> std::same_as<double>;
};

template <typename T>
concept Domain1d = ConstantDomain1d<T> || FunctionalDomain1d<T>;
} // namespace nld
