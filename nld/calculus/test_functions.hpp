#pragma once

#include <nld/calculus/concepts.hpp>
#include <nld/math/segment.hpp>
#include <optional>

#include <nld/calculus/adnum.hpp>

namespace nld {
/// @brief Test functions for calculations.
/// @details We use Ritz method to compute calculus problems,
/// So test and trial functions are same.
/// @tparam Space The space.
/// @tparam Bases The basis what will be used for calculations.
template <typename Space, typename Bases>
struct test_functions final {
    /// @brief Dimension of test functions.
    static constexpr nld::index dimension = std::tuple_size_v<Bases>;
    static constexpr bool with_subdomains = nld::AllWithSubdomains<Bases>;

    /// @brief Create one dimension test functions.
    /// @tparam Basis bases tuple.
    /// @param basis bases tuple.
    template <typename... Basis>
    explicit test_functions(const Space &space, Basis... basis)
        : space(space), bases(std::make_tuple(basis...)), size(0) {
        nld::utils::for_each(
            bases, [this](auto &&basis) { this->size += basis.count(); });
    }

    /// @brief The value compute function of i-th test function
    /// @param i index of test function
    /// @return Function f(x) -> std::array<double, Dim> to compute i-th
    /// function at point x.
    auto value(nld::index i) const
        requires(dimension == 1)
    {
        return [basis = std::get<0>(bases), i](auto x) -> adnum {
            return basis.value(i)(x);
        };
    }

    /// @brief The value compute function of i-th test function
    /// @param i index of test function
    /// @return Function f(x) -> std::array<double, Dim> to compute i-th
    /// function at point x.
    auto value(std::tuple<nld::index> i) const { return value(std::get<0>(i)); }

    /// @brief Get the domain of i-th test function.
    /// @param i index of test function.
    /// @return I-th test function domain.
    auto subdomain(nld::index i) const -> std::optional<nld::segment>
        requires(dimension == 1)
    {
        auto basis = std::get<0>(bases);
        return basis.subdomain(i);
    }

    /// @brief Get the domain of i-th test function.
    /// @param i index of test function.
    /// @return I-th test function domain.
    auto subdomain(std::tuple<nld::index> i) const {
        return subdomain(std::get<0>(i));
    }

    /// @brief Count of test functions for approximation.
    /// @returns Number of test functions.
    auto count() const -> std::tuple<nld::index> { return std::tuple(size); }

    /// @brief Get the space object.
    /// @return Space where test functions are defined.s
    auto get_space() const -> const Space & { return space; }

private:
    const Space &space; ///< Space where we define test functions.
    Bases bases;        ///< Bases in different directions.
    nld::index size;    ///< Count of basis functions in approximation.
};

/// @brief Deduction gide to create test function from basis functions.
template <typename Space, typename... Basis>
test_functions(const Space &space, Basis... basis)
    -> test_functions<Space, std::tuple<Basis...>>;
} // namespace nld
