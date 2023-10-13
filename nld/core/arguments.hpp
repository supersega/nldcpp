#pragma once

namespace nld {

/// @brief Function to pass additional arguments into algorithms.
/// @details This function should be used to pass additional arguments
/// for mathematical algorithm, to avoid lambda wrapping for the call.
/// Note: this function does not extends lifetime.
/// @param args just arguments.
/// @returns Tuple of arguments.
template<typename... Args>
inline constexpr auto arguments(Args&&... args) -> std::tuple<Args&&...> {
    return std::forward_as_tuple(std::forward<Args>(args)...);
}

/// @brief Just function to be called as default argument.
/// @details When we want to note that no additional arguments
/// will be added to algorithm, we use this function to show it.
/// @returns Empty tuple.
inline constexpr auto no_arguments() -> std::tuple<> {
    return arguments();
}

}