#pragma once

#include <nld/calculus/adnum.hpp>

namespace nld {

/// @brief
/// @tparam E
/// @tparam W
/// TODO: Dissallow to make derivative<derivative<E, W1>, W2> and so.
template <typename E, typename W>
struct derivative final {
    using expression_t = std::remove_reference_t<E>;

    /// @brief The derivative order.
    constexpr static auto order_v = std::tuple_size_v<decltype(W::args)>;

    /// @brief Construct a new derivative object
    /// @param e The expression.
    /// @param w With respect to variables.
    explicit derivative(E &&e, W &&w)
        : expression(std::forward<E>(e)), wrt(std::forward<W>(w)) {}

    /// @brief Get derivatives for i-th test function.
    /// @param i coordinate number.
    auto value(std::tuple<nld::index> i) const
        requires(expression_t::dimension == 1)
    {
        return [*this, i](auto x) -> adnum {
            const auto &sp = this->expression.get_space();
            auto &xcoord = sp.coords()[0];
            xcoord = x;
            return autodiff::val(autodiff::derivative<order_v>(
                expression.value(i), wrt, at(xcoord)));
        };
    }

    /// @brief Get the sub domain of derivative expression.
    /// @param i index of test function.
    /// @return I-th test function domain.
    auto subdomain(std::tuple<nld::index> i) const {
        return expression.subdomain(i);
    }

    /// @brief Count of test functions for approximation.
    /// @returns Number of test functions.
    auto count() const -> std::tuple<nld::index> { return expression.count(); }

private:
    E expression;  ///< Calculus expression.
    mutable W wrt; ///< Derivative w.r.t. this parameters will be calculated.
};

template <typename E, typename W>
auto diff(E &&e, W &&w) {
    return derivative<E, W>{std::forward<E>(e), std::forward<W>(w)};
}

} // namespace nld
