#pragma once

#include <nld/core.hpp>
#include <nld/math.hpp>

#include <nld/collocations/collocation_points.hpp>

namespace nld::collocations {

namespace detail {
/// @brief generate Legendre points for a given polynomial polynomial
/// polynomial_degree on the interval [-1, 1]
/// @param degree polynomial degree
/// @return vector with points
inline nld::vector_xd legandre_points(std::size_t degree) {
    nld::vector_xd points(degree);
    switch (degree) {
    case 1:
        points[0] = 0.0;
        break;
    case 2:
        points[0] = -0.5773502691896257;
        points[1] = 0.5773502691896257;
        break;
    case 3:
        points[0] = -0.7745966692414834;
        points[1] = 0.0;
        points[2] = 0.7745966692414834;
        break;
    case 4:
        points[0] = -0.8611363115940526;
        points[1] = -0.3399810435848563;
        points[2] = 0.3399810435848563;
        points[3] = 0.8611363115940526;
        break;
    case 5:
        points[0] = -0.9061798459386640;
        points[1] = -0.5384693101056831;
        points[2] = 0.0;
        points[3] = 0.5384693101056831;
        points[4] = 0.9061798459386640;
        break;
    case 6:
        points[0] = -0.9324695142031521;
        points[1] = -0.6612093864662645;
        points[2] = -0.2386191860831969;
        points[3] = 0.2386191860831969;
        points[4] = 0.6612093864662645;
        points[5] = 0.9324695142031521;
        break;
    case 7:
        points[0] = -0.9491079123427585;
        points[1] = -0.7415311855993945;
        points[2] = -0.4058451513773972;
        points[3] = 0.0;
        points[4] = 0.4058451513773972;
        points[5] = 0.7415311855993945;
        points[6] = 0.9491079123427585;
        break;
    }
    return points;
}

/// @brief generate Legendre weights for a given polynomial polynomial_degree
/// on the interval [-1, 1]
/// @param degree polynomial degree
/// @return vector with weights
inline nld::vector_xd legandre_weights(std::size_t degree) {
    nld::vector_xd weights(static_cast<std::size_t>(degree));
    switch (degree) {
    case 1:
        weights[0] = 2.0;
        break;
    case 2:
        weights[0] = 1.0;
        weights[1] = 1.0;
        break;
    case 3:
        weights[0] = 0.5555555555555556;
        weights[1] = 0.8888888888888888;
        weights[2] = 0.5555555555555556;
        break;
    case 4:
        weights[0] = 0.3478548451374538;
        weights[1] = 0.6521451548625461;
        weights[2] = 0.6521451548625461;
        weights[3] = 0.3478548451374538;
        break;
    case 5:
        weights[0] = 0.2369268850561891;
        weights[1] = 0.4786286704993665;
        weights[2] = 0.5688888888888889;
        weights[3] = 0.4786286704993665;
        weights[4] = 0.2369268850561891;
        break;
    case 6:
        weights[0] = 0.1713244923791704;
        weights[1] = 0.3607615730481386;
        weights[2] = 0.4679139345726910;
        weights[3] = 0.4679139345726910;
        weights[4] = 0.3607615730481386;
        weights[5] = 0.1713244923791704;
        break;
    case 7:
        weights[0] = 0.1294849661688697;
        weights[1] = 0.2797053914892766;
        weights[2] = 0.3818300505051189;
        weights[3] = 0.4179591836734694;
        weights[4] = 0.3818300505051189;
        weights[5] = 0.2797053914892766;
        weights[6] = 0.1294849661688697;
        break;
    }
    return weights;
}

inline void transform_to_interval(nld::vector_xd &nodes,
                                  nld::segment interval) {
    // from interval [-1, 1] to interval [a, b]
    auto h = interval.end - interval.begin;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        nodes[i] = interval.begin + (nodes[i] + 1.0) * h / 2.0;
    }
}
} // namespace detail

/// @brief Build Legendre collocation points on a given segment with a given
/// count of points
/// @param segment interval
/// @param degree polynomial degree
inline auto legandre_collocation_points(nld::segment interval,
                                        std::size_t degree) -> nld::vector_xd {
    auto m = degree;
    auto points = detail::legandre_points(m);
    detail::transform_to_interval(points, interval);
    return points;
}
} // namespace nld::collocations
