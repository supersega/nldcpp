#pragma once

namespace nld {
/// @brief The integral type.
template<typename E, typename D>
struct integral final {
    /// @brief Construct a new integral object.
    /// @param e expression.
    explicit integral(E&& e, D&& d) : expression(std::forward<E>(e)), domain(std::forward<D>(d)) { }

    E expression; ///< The underlying expression.
    D domain;     ///< The integration domain.
};

template<typename E, typename D>
integral(E&&, D&&) -> integral<E, D>;
}