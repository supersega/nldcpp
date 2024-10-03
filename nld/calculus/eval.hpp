#pragma once

#include <nld/calculus/concepts.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/calculus/tensor_size.hpp>

namespace nld::internal {
template <typename I, typename E, typename D, typename O>
auto eval_impl(const E &mul, const D &domain, O options)
    requires(tensor_size<E>::size == 1)
{
    auto [ni] = mul.count();
    nld::tensor<1> result(ni);
    for (nld::index i = 0; i < ni; i++)
        result(i) = nld::integrate<I>(
            [i, &mul](auto... x) {
                return static_cast<double>(mul.value(std::tuple(i))(x...));
            },
            domain, options);

    return result;
}

template <typename I, typename E, typename F, typename D, typename O>
auto eval_impl(const nld::dirac_shift<E, F> &shift, const D &domain,
               [[maybe_unused]] O options) {
    auto [ni] = shift.expression.count();
    nld::tensor<1> result(ni);
    for (nld::index i = 0; i < ni; i++)
        result(i) = static_cast<double>(
            std::apply(shift.expression.value(std::tuple(i)),
                       shift.delta_function.coords()));

    return result;
}

template <typename I, typename E, typename D, typename O>
auto eval_impl(const E &mul, const D &domain, O options)
    requires(tensor_size<E>::size == 2)
{
    auto [ni, nj] = mul.count();
    nld::tensor<2> result(nj, ni);
    for (nld::index i = 0; i < ni; i++)
        for (nld::index j = 0; j < nj; j++) {
            if constexpr (SubdomainDefined<E>) {
                nld::segment domain_{domain.begin, domain.end};
                auto subdomain = mul.subdomain(std::tuple(j, i));
                subdomain =
                    subdomain ? domain_.intersect(*subdomain) : std::nullopt;
                result(j, i) =
                    subdomain ? nld::integrate<I>(
                                    [j, i, &mul](auto... x) {
                                        return static_cast<double>(
                                            mul.value(std::tuple(j, i))(x...));
                                    },
                                    *subdomain, options)
                              : 0.0;
            } else {
                result(j, i) = nld::integrate<I>(
                    [j, i, &mul](auto... x) {
                        return static_cast<double>(
                            mul.value(std::tuple(j, i))(x...));
                    },
                    domain, options);
            }
        }
    return result;
}

template <typename I, typename E, typename D, typename O>
auto eval_impl(const E &expr, const D &domain, O options)
    requires(tensor_size<E>::size == 3)
{}

template <typename I, typename E, typename D, typename O>
auto eval_impl(const E &expr, const D &domain, O options)
    requires(tensor_size<E>::size == 4)
{}

template <typename I, typename E, typename D, typename O>
auto eval(const E &expr, const D &domain, O options)
    -> nld::tensor<tensor_size<E>::size> {
    return eval_impl<I>(expr, domain, options);
}

} // namespace nld::internal
