#pragma once

#include <algorithm>
#include <nld/math/constants.hpp>
#include <optional>

namespace nld {

namespace detail {
inline auto are_close(double a, double b, double tolerance) -> bool {
    return std::abs(a - b) < tolerance;
}
} // namespace detail

/// @brief 1d segment representation.
struct segment final {
    /// @brief Intersect two intervals.
    /// @param other The other interval.
    /// @return some interval if intervals intersect, nullopt otherwise
    auto intersect(const segment &other) const -> std::optional<segment> {
        if (other.begin > end || begin > other.end)
            return std::nullopt;

        if (detail::are_close(end, other.begin, nld::GEOMETRY_TOLERANCE) ||
            detail::are_close(begin, other.end, nld::GEOMETRY_TOLERANCE))
            return std::nullopt;

        return segment{std::max(begin, other.begin), std::min(end, other.end)};
    }

    /// @brief Infinity segment.
    static auto infinity() -> nld::segment {
        return segment{-1000000.0, 1000000.0};
    }

    double begin; ///< The begin of the segment.
    double end;   ///< The end of the segment.
};
} // namespace nld
