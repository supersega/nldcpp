#pragma once

#include <nld/calculus/adnum.hpp>
#include <nld/calculus/tensor_size.hpp>

namespace nld {

/// @brief The mul expression holds left and right nodes.
template<typename L, typename R>
struct mul final {
    L left;  ///< Left operand.
    R right; ///< Right operand.
};

/// @brief The product expression holds left and right nodes.
template<typename L, typename R>
struct tensor_mul final {
    using left_t = std::remove_reference_t<L>;
    using right_t = std::remove_reference_t<R>;

    /// @brief Function what computes value of tensor expression.
    /// @param indicies Tensor indicies.
    /// @return Function what computes value of tensor expression.
    template<typename... Dims>
    auto value(std::tuple<Dims...> indicies) const {
        return [this, indicies](auto x) -> adnum {
            if constexpr (Constant<left_t>) {
                auto r = this->right.value(nld::utils::tail_view<nld::tensor_size<right_t>::size>(indicies));
                return this->left * r(x);
            }

            if constexpr (Constant<right_t>) {
                auto l = this->left.value(nld::utils::head_view<nld::tensor_size<left_t>::size>(indicies));
                return this->right * l(x);
            }

            if constexpr (!Constant<left_t> && !Constant<right_t>) {
                auto l = this->left.value(nld::utils::head_view<nld::tensor_size<left_t>::size>(indicies));
                auto r = this->right.value(nld::utils::tail_view<nld::tensor_size<right_t>::size>(indicies));
                return l(x) * r(x);
            }
        };
    }

    /// @brief Get the domain of i-th test function.
    /// @param i index of test function.
    /// @return I-th test function domain.
    template<typename... Dims>
    auto subdomain(std::tuple<Dims...> i) const -> std::optional<nld::segment> {
        if constexpr ((Constant<left_t> && ScalarFunctionExpression<right_t>) || (Constant<right_t> && ScalarFunctionExpression<left_t>))
            return nld::segment::infinity();
        else if constexpr (Constant<left_t> || ScalarFunctionExpression<left_t>)
            return right.subdomain(nld::utils::tail_view<nld::tensor_size<right_t>::size>(i));
        else if constexpr (Constant<right_t> || ScalarFunctionExpression<right_t>)
            return left.subdomain(nld::utils::head_view<nld::tensor_size<left_t>::size>(i));
        else if constexpr (!(Constant<left_t> || ScalarFunctionExpression<left_t>) && !(Constant<right_t> || ScalarFunctionExpression<right_t>)) {
            auto l = left.subdomain(nld::utils::head_view<nld::tensor_size<left_t>::size>(i));
            auto r = right.subdomain(nld::utils::tail_view<nld::tensor_size<right_t>::size>(i));
            if (!l || !r)
                return std::nullopt;
            
            return l->intersect(*r);
        }
    }

    /// @brief Count of elements in mul expression for approximation.
    /// @returns Number of test functions.
    auto count() const {
        if constexpr (Constant<left_t> || ScalarFunctionExpression<left_t>)
            return std::tuple(right.count());
        else if constexpr (Constant<right_t> || ScalarFunctionExpression<right_t>)
            return std::tuple(left.count());
        else if constexpr (!(Constant<left_t> || ScalarFunctionExpression<left_t>) && !(Constant<right_t> || ScalarFunctionExpression<right_t>))
            return std::tuple_cat(std::tuple(left.count()), std::tuple(right.count()));
    }

    L left;  ///< Left operand.
    R right; ///< Right operand.
};

/// @brief The Dirac shift expression expression holds left and right nodes.
template<typename E, typename D>
struct dirac_shift final {
    /// @brief Construct a new dirac shift object
    /// @param e expression.
    /// @param d delta fuction.
    explicit dirac_shift(E&& e, D&& d) : expression(std::forward<E>(e)), delta_function(std::forward<D>(d)) { }
    
    E expression;     ///< The expression.
    D delta_function; ///< The delta function.
};

}
