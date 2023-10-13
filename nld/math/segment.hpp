#pragma once

#include <optional>

namespace nld {

/// @brief 1d segment representation.
struct segment final {
    /// @brief Intersect two intervals.
    /// @param other The other interval. 
    /// @return some interval if intervals intersect, nullopt otherwise 
    auto intersect(const segment& other) const -> std::optional<segment> {
        if (other.begin > end || begin > other.end)
            return std::nullopt;
        
        return segment { std::max(begin, other.begin), std::min(end, other.end) };
    }

    /// @brief Infinity segment.
    static auto infinity() -> nld::segment {
        return segment { -1000000.0, 1000000.0 };
    }

    double begin; ///< The begin of the segment.
    double end;   ///< The end of the segment.
};
}