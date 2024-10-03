#pragma once

#include <functional>
#include <nld/core.hpp>
#include <nld/math/concepts.hpp>

namespace nld {
struct constant_domain final {
    double begin; ///< The begin of the domain.
    double end;   ///< The end of the domain.
};

struct variable_domain final {
    std::function<double(double)> begin; ///< The begin of the domain.
    std::function<double(double)> end;   ///< The end of the domain.
};
} // namespace nld
