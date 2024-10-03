#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// std includes
#include <cmath>
#include <iostream>
using namespace std;

// nld includes
#include <nld/math.hpp>
using namespace nld;

TEST_CASE("Require that integrate with gauss_kronrod21 traits calculates "
          "integral correctly for linear function: R -> R",
          "[nld::integrate<nld::quad>()]") {
    REQUIRE(integrate<gauss_kronrod21>(
                [](auto x) { return x; }, segment{0.0, 1.0},
                nld::gauss_kronrod21::integration_options{}) ==
            Catch::Approx(0.5));
}
