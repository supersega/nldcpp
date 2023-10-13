#pragma once

namespace nld {

/// @brief Make integration arguments for problem T.
/// @details This function should be overloaded for special 
/// type like: periodic problem, or bifurcation continuation
/// problem (see saddle_node for example).
/// @param variables problem variables (initial condition, other parameters)
/// @param problem problem we want to compute integration arguments
/// @return tuple with (initial conditions, arguments as tuple)
template<typename T, typename V>
auto integration_arguments(const V& variables, const T& problem) = delete;

} /// end namespace nld