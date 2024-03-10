#include <catch.hpp>

// nld includes
#include <nld/core.hpp>
using namespace nld;

#include <iostream>

SCENARIO("Tuple for_each", "[for_each]") {
    GIVEN("tuple with some elements") {
        auto t = std::tuple(2.0, -1, 1u);
        WHEN("increment called for all elements tuple is changed") {
            nld::utils::for_each(t, [ ](auto& e) { e++; });
            std::tuple expected(3.0, 0, 2u);
            REQUIRE(t == expected);
        }
    }
    GIVEN("tuple with references") {
        auto a = 1.0;
        auto b = 2u;
        auto c = -3;
        auto t = std::tie(a, b, c);
        WHEN("for_each change elements, references are changed") {
            nld::utils::for_each(t, [ ](auto& e) { e = 0; });
            REQUIRE((a == Catch::Approx(0.0) && b == 0 && c == 0));
        }
    }
    GIVEN("empty tuple") {
        auto t = std::tuple();
        WHEN("call for_each for empty tuple - function never called") {
            auto calls = 0u;
            nld::utils::for_each(t, [&calls](auto) { calls++; });
            REQUIRE(calls == 0u);
        }
    }
}
