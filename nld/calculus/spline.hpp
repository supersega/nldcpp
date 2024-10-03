#pragma once

#include <nld/calculus/space.hpp>

namespace nld::internal {
/// @brief Compute cubic B-Spline on interval.
/// @param tau Value between begin end end.
/// @param begin Begin of the interval.
/// @param end End of the interval.
/// @return Value of cubic B-Spline at tau.
inline auto spline3(adnum tau, adnum begin, adnum end) -> adnum {
    adnum length = end - begin;
    adnum x = 4.0 * (tau - begin) / length - 2.0;
    if (x <= -2.0)
        return 0.0;
    if (x <= -1.0)
        return 0.25 * (x + 2.0) * (x + 2.0) * (x + 2.0);
    if (x <= 0.0)
        return -0.75 * x * x * x - 1.5 * x * x + 1.0;
    if (x <= 1.0)
        return 0.75 * x * x * x - 1.5 * x * x + 1.0;
    if (x <= 2.0)
        return -0.25 * x * x * x + 1.5 * x * x - 3.0 * x + 2.0;
    return 0.0;
}

/// @brief Compute cubic B-Spline 1-st derivative on interval.
/// @param tau Value between begin end end.
/// @param begin Begin of the interval.
/// @param end End of the interval.
/// @return 1-st derivative of cubic B-Spline at tau.
inline auto dspline3(double tau, double begin, double end) -> double {
    double length = end - begin;
    double x = 4.0 * (tau - begin) / (length)-2.0;
    double dxdtau = 4.0 / length;
    if (x <= -2.0)
        return 0.0;
    if (x <= -1.0)
        return (3.0 * 0.25 * (x + 2.0) * (x + 2.0)) * dxdtau;
    if (x <= 0.0)
        return (-3.0 * 0.75 * x * x - 3.0 * x) * dxdtau;
    if (x <= 1.0)
        return (3.0 * 0.75 * x * x - 3.0 * x) * dxdtau;
    if (x <= 2.0)
        return (-3.0 * 0.25 * x * x + 3.0 * x - 3.0) * dxdtau;
    return 0.0;
}

/// @brief Compute cubic B-Spline 1-st derivative on interval.
/// @param tau Value between begin end end.
/// @param begin Begin of the interval.
/// @param end End of the interval.
/// @return 2-nd derivative of cubic B-Spline at tau.
inline auto ddspline3(double tau, double begin, double end) -> double {
    double length = end - begin;
    double x = 4.0 * (tau - begin) / (length)-2.0;
    double dxdtau = 4.0 / length;
    if (x <= -2.0)
        return 0.0;
    if (x <= -1.0)
        return (6.0 * 0.25 * (x + 2.0)) * dxdtau * dxdtau;
    if (x <= 0.0)
        return (-6.0 * 0.75 * x - 3.0) * dxdtau * dxdtau;
    if (x <= 1.0)
        return (6.0 * 0.75 * x - 3.0) * dxdtau * dxdtau;
    if (x <= 2.0)
        return (-6.0 * 0.25 * x + 3.0) * dxdtau * dxdtau;
    return 0.0;
}
} // namespace nld::internal
