#pragma once
#include <cstddef>

namespace nld {
/// @brief Parameters for periodic solution continuation.
struct periodic_parameters {
  explicit periodic_parameters(std::size_t p, std::size_t i)
      : periods(p), intervals(i) {}

  std::size_t periods;   ///< number of periods.
  std::size_t intervals; ///< number of integration intervals.
};
} // namespace nld
