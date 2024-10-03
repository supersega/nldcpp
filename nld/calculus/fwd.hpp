#pragma once

#include <nld/math.hpp>

namespace nld {
template <typename L, typename R>
struct add;

template <typename L, typename R>
struct mul;

template <typename L, typename R>
struct tensor_mul;

template <typename Space, typename Basis>
struct test_functions;

template <typename T, typename B>
struct weighted_test_functions;

template <typename E, typename W>
struct derivative;

template <typename T>
struct eigenfunctions;

template <typename F>
struct scalar_function;

template <typename E, typename D>
struct integral;

template <typename P>
struct delta_function;

template <typename E, typename D>
struct dirac_shift;

struct boundary_condition;

struct basis;

template <typename E, typename I, FunctionalDomain1d D, typename O>
struct variable_integral;
} // namespace nld
