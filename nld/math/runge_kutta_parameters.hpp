#pragma once
namespace nld::math {

/// @brief Container for ODE constant step integration parameter.
/// @tparam Float floating point.
struct constant_step_parameters final {
    /// Constructor just to deduce template arguments.
    /// @param start Start integration time.
    /// @param end End integration time.
    /// @param intervals Number of integration intervals.
    constant_step_parameters(double start, double end, std::size_t intervals) :
        start{ start }, end{ end }, intervals{ intervals } {
    };

    double start;           ///< start integration time.
    double end;             ///< end integration time.
    std::size_t intervals; ///< number of integration intervals.
};
} // end namespace nld::math

namespace nld {
    using nld::math::constant_step_parameters;
}
