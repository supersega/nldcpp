#pragma once

#include <nld/math.hpp>

namespace nld {

/// @brief Enumeration for continuation direction
enum class direction : int {
    forward = 1, ///< direction to explicit parameter increase.
    reverse = -1 ///< direction to explicit parameter decrease.
};

/// @brief Structure to hold continuation parameters.
/// @details For now we have constant arc length step, we may change this 
/// in future version of nld.
/// @tparam F floating point type.
struct continuation_parameters final {
    /// @brief ctor just to deduce template argument.
    /// @param newton_parameters parameters for Newton method.
    /// @param total_param_length total param length.
    /// @param param_min_step param min step.
    /// @param param_max_step param min step.
    /// @param direction direction for continuation algorithm.
    continuation_parameters(
        nld::newton_parameters newton_parameters,
        double total_param_length,
        double param_min_step,
        double param_max_step,
        nld::direction direction) :
        newton_parameters{ newton_parameters },
        total_param_length{ total_param_length }, 
        param_min_step{ param_min_step },
        param_max_step{ param_max_step },
        direction{ direction } { }

    nld::newton_parameters newton_parameters; ///< parameters for Newton method.
    double total_param_length;                ///< total param length.
    double param_min_step;                    ///< param min step.
    double param_max_step;                    ///< param min step.
    nld::direction direction;                 ///< direction for continuation algorithm.
};
} /// end namespace nld
