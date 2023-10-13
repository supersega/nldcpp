#pragma once

#include <nld/math.hpp>

#include <nld/calculus/basis.hpp>
#include <nld/calculus/spline.hpp>

namespace nld {
/// @brief Basis function represented by cubic B-Spline.
/// @details This is a class what represents a cubic B-Spline
/// basis in function space.
/// @see https://en.wikipedia.org/wiki/B-spline.
struct bspline_3_basis final : basis {
    /// @brief Use ctor of the base class.
    using basis::basis;

    /// @brief The value of the basis.
    /// @param i Index of the basis function.
    /// @return Callable value i-th basis function.
    auto value(nld::index i) const {
        auto [a, b] = interval(i);
        return [a = a, b = b](auto x) -> adnum {
            return nld::internal::spline3(x, a, b);
        };
    }

    /// @brief Get domain where i-th basis function is defined.
    /// @param i function index.
    /// @return nld::segment where i-th basis function is defined.
    auto subdomain(nld::index i) const -> nld::segment {
        auto [a, b] = interval(i);
        auto subdomain = nld::segment{a, b};
        return *subdomain.intersect(basis::domain());
    }

private:
    auto interval(nld::index idx) const -> std::tuple<double, double> {
        auto dx = (definition.interval.second - definition.interval.first) /
                  (definition.count - 3);
        auto start = -2 * dx + (idx - 1) * dx;
        auto stop = 2 * dx + (idx - 1) * dx;
        return std::tuple(start, stop);
    }
};
} // namespace nld
