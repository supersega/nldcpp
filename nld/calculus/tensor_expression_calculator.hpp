#pragma once

#include <nld/calculus/eval.hpp>
#include <nld/calculus/join.hpp>

#include <nld/math.hpp>

#include <tuple>

namespace nld {

/// @brief Calculate calculus expression.
template<typename I>
struct tensor_expression_calculator final {
    explicit tensor_expression_calculator(integration_options opt = integration_options{}) : options(opt) { }

    template <typename E, typename D>
    auto calculate(const integral<E, D>& ie) const {
        std::tuple<> t;
        return calculate(ie.expression, ie.domain, t);
    }

private:
    template<typename L, typename R, typename D, typename T>
    auto calculate(const nld::add<L, R>& add, const D& domain, const T& tuple) const {
        auto left = calculate(add.left, domain, tuple);
        return calculate(add.right, domain, left);
    }

    template<typename L, typename R, typename D, typename T>
    auto calculate(const nld::mul<L, R>& mul, const D& domain, const T& tuple) const {
        auto left = calculate(mul.left);
        auto right = calculate(mul.right, domain, std::make_tuple());

        static_assert(std::tuple_size_v<decltype(left)> == 1 && std::tuple_size_v<decltype(right)> == 1, 
            "It is not possible now to use complicated expression in internal integral");

        auto joined = join(std::get<0>(left), std::get<0>(right));
        return std::tuple_cat(tuple, std::tuple(joined));
    }

    template<typename E, typename D, typename T>
    auto calculate(const E& mul, const D& domain, const T& tuple) const {
        return std::tuple_cat(tuple, std::tuple(nld::internal::eval<I>(mul, domain, options)));
    }

    template<typename E, typename F, typename D, typename T>
    auto calculate(const dirac_shift<E, F>& mul, const D& domain, const T& tuple) const {
        return std::tuple_cat(tuple, std::tuple(nld::internal::eval<I>(mul, domain, options)));
    }
private:
    integration_options options;
};

}
