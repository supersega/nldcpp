#pragma once

#include <nld/systems/jacobian_mixin.hpp>

#include <utility>
namespace nld {

/// @brief bind a function to a fixed coordinate of the unknown vector
/// @details This class is used to bind a function to a fixed coordinate
/// of the unknown vector. It suppose to be used in
/// fastest_coordinate_continuation. Also it is a good choise to use it for
/// initial guess for autonous systems periodic solutions continuation.
/// @tparam Fn the type of the function to be bound.
template <typename Fn>
struct bind_wrt_unknown : nld::jacobian_mixin<bind_wrt_unknown<Fn>> {
    /// @brief bind function to a fixed coordinate of the unknown vector
    /// @param ds the function to be bound
    bind_wrt_unknown(Fn &&fn, std::size_t coord, double value, std::size_t dim)
        : function{std::forward<Fn>(fn)}, coordinate{coord}, value{value},
          dim{dim} {}

    /// @brief operator() to be used by The Newton method
    /// @param u the unknown vector [u_1, u2, ..., u_coord - 1, u_coord + 1,
    /// ..., u_dim + 1]
    template <typename Vector>
    auto operator()(const Vector &u) const -> Vector {
        nld::vector_xdd u_merged = merged_unknown(u);
        return function(u_merged);
    }

    /// @brief merge the unknown vector with the value of the fixed coordinate
    /// @param u the unknown vector [u_1, u2, ..., u_coord - 1, u_coord + 1,
    /// ..., u_dim + 1]
    /// @return the merged vector [u_1, u2, ..., u_coord - 1, value, u_coord +
    /// 1,
    /// ..., u_dim + 1]
    template <typename Vector>
    auto merged_unknown(const Vector &u) const -> Vector {
        Vector u_merged(dim + 1);
        u_merged << u.head(coordinate), value, u.tail(dim - coordinate);
        return u_merged;
    }

private:
    Fn function;
    std::size_t coordinate;
    double value;
    std::size_t dim;
};

template <typename F>
bind_wrt_unknown(F &&function, std::size_t coord, double value, std::size_t dim)
    -> bind_wrt_unknown<F>;

} // namespace nld
