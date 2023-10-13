#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// std includes
#include <cmath>
#include <iostream>
using namespace std;

// nld includes
#include <nld/math.hpp>
using namespace nld;

#include "utils.hpp"

SCENARIO("Newton method on simple equation", "[newton]") {
    GIVEN("simple equation") {
        auto f = [ ](dual x, dual a) {
            vector_xdd y(1);
            y << x * x - a;
            return y;
        };
        WHEN("iterations and tolerance enough") {
            constexpr auto max_iterations = 10;
            constexpr auto tolerance = 1.0e-5;

            THEN("Newton method is convergence") {
                dual x = 1.9;
                auto info = newton(f, wrt(x), at(x, 4.0), newton_parameters(max_iterations, tolerance));
                REQUIRE((info.is_convergence && info.number_of_done_iterations < max_iterations));
            }
        }
        WHEN("iterations not enough and tolerance to small") {
            constexpr auto max_iterations = 2;
            constexpr auto tolerance = 1.0e-8;
            
            THEN("Newton method is not convergence") {
                dual x = 1.9;
                auto info = newton(f, wrt(x), at(x, 4.0), newton_parameters(max_iterations, tolerance));
                REQUIRE((!info.is_convergence && info.number_of_done_iterations == max_iterations));
            }
        }
    }
}

SCENARIO("Runge-Kutta method on simple equation", "[runge_kutta_4]") {
    GIVEN("differential equation and initial conditions") {
        // this ode is from https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php
        auto ode = [ ](vector_xd y, double x) {
            vector_xd dy(1);
            dy(0) = (5 * x * x - y(0)) / exp(x + y(0));
            return dy;
        };
        vector_xd y0 = vector_xd::Ones(1);
        WHEN("solve ode using Runge-Kutta method") {
            constexpr auto intervals = 10;

            vector_xd expected_solution(intervals + 1);
            expected_solution << 1.0, 0.9655827899, 0.937796275, 0.9189181059, 0.9104421929,
            0.913059839, 0.9267065986, 0.9506796142, 0.9838057659, 1.024628046, 1.0715783953;
            THEN("solution is approximated as we expect") {
                auto sln = runge_kutta_4::solution(ode, constant_step_parameters{ 0.0, 1.0, intervals }, y0);

                REQUIRE((sln.rows() == intervals + 1 && sln.cols() == 1));
                REQUIRE(are_equal(expected_solution, sln.col(0)));
            }
            AND_THEN("max and min returns an expected value") {
                REQUIRE(max<runge_kutta_4>(ode, constant_step_parameters{ 0.0, 1.0, intervals }, y0)(0) == Approx(expected_solution.maxCoeff()));
                REQUIRE(min<runge_kutta_4>(ode, constant_step_parameters{ 0.0, 1.0, intervals }, y0)(0) == Approx(expected_solution.minCoeff()));
            }
            AND_THEN("end_solution returns expected result") {
                REQUIRE(runge_kutta_4::end_solution(ode, constant_step_parameters{ 0.0, 1.0, intervals }, y0)(0) == Approx(expected_solution(intervals)));
            }
            AND_THEN("half_swing returns expected value") {
                REQUIRE(mean<runge_kutta_4>(ode, constant_step_parameters{ 0.0, 1.0, intervals }, y0)(0) == Approx(0.5 * (expected_solution.maxCoeff() - expected_solution.minCoeff())));
            }
        }
    }
}

TEST_CASE("Require that quad integration calculates integral for F: R -> R") {

}