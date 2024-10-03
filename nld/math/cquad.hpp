#pragma once

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include <functional>
#include <memory>
#include <nld/core.hpp>

#include <nld/math/concepts.hpp>
#include <nld/math/segment.hpp>

namespace nld {

/// @brief The 1-d quad integration.
/// @details The quad integration based on GSL library.
/// GSL has a good quality of integration, so it is better
/// to use 3rd party for such things.
struct cquad final {
    /// @brief Integration options for cquad.
    struct integration_options {
        double absolute_tolerance =
            1.49e-8; ///< Absolute integration tolerance.
        double relative_tolerance =
            1.49e-8;               ///< Relative integration tolerance.
        std::size_t intervals = 3; ///< Limit of intervals.
    };

    /// @brief The integration function.
    /// @tparam F The function type to intergate.
    /// @tparam T The tuple of additional arguments.
    /// @param function The function to integrate.
    /// @param domain The domain to integrate.
    /// @param options An integration options.
    /// @param args The additional arguments to function.
    /// @return The value of the integral.
    template <typename F, Domain1d D, typename T>
    static auto integrate(F function, D domain,
                          cquad::integration_options options, T &&args) {
        using workspace = std::unique_ptr<
            gsl_integration_cquad_workspace,
            std::function<void(gsl_integration_cquad_workspace *)>>;
        struct context {
            F function;
            T &&args;
        };

        auto gsl_callable = [](double x, void *params) -> double {
            auto &ctx = *(reinterpret_cast<context *>(params));
            return std::apply(ctx.function,
                              std::tuple_cat(nld::arguments(x), ctx.args));
        };

        gsl_function gsl_f;
        gsl_f.function = gsl_callable;
        auto ctx = context{std::move(function), std::forward<T>(args)};
        gsl_f.params = std::addressof(ctx);

        double result, error;
        std::size_t neval;
        workspace ws(gsl_integration_cquad_workspace_alloc(options.intervals),
                     gsl_integration_cquad_workspace_free);
        gsl_integration_cquad(
            &gsl_f, domain.begin, domain.end, options.absolute_tolerance,
            options.relative_tolerance, ws.get(), &result, &error, &neval);

        return result;
    }
};
} // namespace nld
