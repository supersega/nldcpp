#pragma once

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include <nld/core.hpp>

#include <nld/math/integration_options.hpp>
#include <nld/math/segment.hpp>

namespace nld {

/// @brief The 1-d quad integration.
/// @details The quad integration based on GSL library.
/// GSL has a good quality of integration, so it is better
/// to use 3rd party for such things.
struct gauss_kronrod21 final {
    /// @brief The integration function.
    /// @tparam F The function type to intergate.
    /// @tparam T The tuple of additional arguments.
    /// @param function The function to integrate.
    /// @param domain The domain to integrate.
    /// @param options An integration options.
    /// @param args The additional arguments to function.
    /// @return The value of the integral. 
    template<typename F, typename T>
    static auto integrate(F function, segment domain, integration_options options, T&& args) {
        using workspace = std::unique_ptr<gsl_integration_workspace, std::function<void(gsl_integration_workspace *)>>;
        struct context { F function; T&& args; };

        auto gsl_callable = [](double x, void* params) -> double {
            auto& ctx = *(reinterpret_cast<context*>(params));
            return std::apply(ctx.function, std::tuple_cat(nld::arguments(x), ctx.args));
        };
        
        gsl_function gsl_f;
        gsl_f.function = gsl_callable;
        auto ctx = context { std::move(function), std::forward<T>(args) };
        gsl_f.params = std::addressof(ctx);

        double result, error;
        workspace ws(gsl_integration_workspace_alloc(options.max_iterations), gsl_integration_workspace_free);
        gsl_integration_qags(&gsl_f, domain.begin, domain.end,
            options.absolute_tolerance, options.relative_tolerance, options.max_iterations,
            ws.get(), &result, &error);

        return result;
    }
};
}
