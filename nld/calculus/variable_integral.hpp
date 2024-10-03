#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/calculus/adnum.hpp>

namespace nld {

/// @brief The integral with variable bounds represented as a function.
template <typename E, typename I, FunctionalDomain1d D, typename O>
struct variable_integral final {
    using expression_t = std::remove_reference_t<E>;

    /// @brief Tensor size of integral expression.
    static constexpr nld::index tensor_size = expression_t::tensor_size;

    /// @brief Construct a new integral object.
    /// @param e expression.
    explicit variable_integral(E &&e, I &&i, D &&d, O &&o)
        : expression(std::forward<E>(e)), integrator(std::forward<I>(i)),
          domain(std::forward<D>(d)), options(std::forward<O>(o)) {}

    /// @brief Get derivatives for i-th test function.
    /// @param i coordinate number.
    auto value(auto i) const {
        return [*this, i](auto x) -> adnum {
            nld::constant_domain domain_{domain.begin(x), domain.end(x)};
            auto value = integrator.integrate(
                [i, &domain_, this](auto... x) {
                    return static_cast<double>(expression.value(i)(x...));
                },
                domain_, options, nld::no_arguments());
            return value;
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
    auto count() const { return expression.count(); }

private:
    E expression; ///< The underlying expression.
    I integrator; ///< The integrator.
    D domain;     ///< The integration domain.
    O options;    ///< The integration options.
};

template <typename E, typename I, FunctionalDomain1d D, typename O>
variable_integral(E &&, I &&, D &&, O &&) -> variable_integral<E, I, D, O>;

} // namespace nld
