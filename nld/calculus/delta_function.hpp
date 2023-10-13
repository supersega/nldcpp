#pragma once

namespace nld {
/// @brief Dirac delta function.
template<typename P>
struct delta_function final {
    /// @brief Construct a new delta function object 
    /// @tparam Args 
    /// @param coords The coords of Dirac Delta function.
    template<typename... Args>
    explicit delta_function(Args... coords) : point(std::make_tuple(coords...)) {

    }

    /// @brief Get coords of Dirac Delta function.
    /// @return The coords of Dirac Delta function.
    auto coords() const -> P {
        return point;
    }
private:
    P point; ///< The coords of Dirac Delta function.
};

template<typename... Args>
delta_function(Args... coords) -> delta_function<std::tuple<Args...>>;
}