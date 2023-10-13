#include <catch.hpp>

#include <cmath>
#include <iostream>
using namespace std;

#include <nld/autocont.hpp>
#include <nld/core.hpp>

using namespace nld;

struct fake_solver {
    using integration_parameters_t = constant_step_parameters;
    
    template<typename... Args>
    explicit fake_solver(Args...) { }
    

    template<typename DS, Vector Reals, typename T = std::tuple<>>
    static Reals end_solution(const DS&, const constant_step_parameters&, const Reals& variables, T&& args = no_arguments()) {
        return variables;
    }

    template<typename DS, Vector Reals, typename T = std::tuple<>>
    static nld::matrix_xd solution(const DS&, const constant_step_parameters&, const Reals&, T&& args = no_arguments()) {
        nld::matrix_xd m(2, 2);
        m << 0, 0,
             3, 0;
        return m;
    }
};

auto fn(const nld::vector_xdd& u, nld::dual lambda)
{
    nld::vector_xdd f(2);

    f[0] = (1.0 - lambda) * u[0] - u[1];
    f[1] = u[0] + u[0] * u[0];

    return f;
}

TEST_CASE("solution mapper just propagate second argument", "[nld::solution()]") {
    const auto solution = nld::solution();
    const nld::vector_xd expected = nld::vector_xd::Ones(2);

    REQUIRE(solution(3, expected) == expected);
}

TEST_CASE("point2d mapper makes point from given indices", "[nld::point2d()]") {
    const auto point2d = nld::point2d(3, 0);

    nld::vector_xd data(4);
    data << 1.0, 2.0, 3.0, 4.0;

    REQUIRE(point2d(3, data) == std::tuple(4.0, 1.0));
}

TEST_CASE("mean_amplitude mapper returns half swing from solver", "[nld::mean_amplitude()]") {
    nld::vector_xd data(4);
    data << 1.0, 2.0, 3.0, 4.0;
    
    auto periodic = nld::periodic<fake_solver>(nld::autonomous(fn), nld::periodic_parameters{ 1, 100 });
    
    // For autonomous system we have arguments (state, T, lambda)
    const auto mean_amplitude = nld::mean_amplitude(1);
    
    REQUIRE(mean_amplitude(periodic, data) == 0.0);
}
