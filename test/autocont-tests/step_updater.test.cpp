#include <catch.hpp>

#include <cmath>
#include <iostream>
using namespace std;

#include <nld/autocont.hpp>
#include <nld/core.hpp>

TEST_CASE("step_updater increase_step_if_possible does not increase step if it is not enough iterations ", "[nld::step_updater]") {
    nld::step_updater updater(nld::step_updater_parameters{ 0.05, 0.1, 2.0, 3 });

    auto expected_step = updater.step();
    updater.increase_step_if_possible();

    REQUIRE(updater.step() == Catch::Approx(expected_step));
}

TEST_CASE("step_updater increase_step_if_possible increasies step if it is enough iterations ", "[nld::step_updater]") {
    nld::step_updater updater(nld::step_updater_parameters{ 0.05, 0.1, 2.0, 2 });

    auto expected_step = updater.step();
    updater.increase_step_if_possible();
    updater.increase_step_if_possible();
    updater.increase_step_if_possible();

    REQUIRE(updater.step() == Catch::Approx(2.0 * expected_step));
}

TEST_CASE("step_updater decrease_step decreases step if it is not to small ", "[nld::step_updater]") {
    nld::step_updater updater(nld::step_updater_parameters{ 0.05, 0.1, 2.0, 0 });

    auto expected_step = updater.step();
    updater.increase_step_if_possible();

    REQUIRE(updater.decrease_step());
    REQUIRE(updater.step() == Catch::Approx(expected_step));
}

TEST_CASE("step_updater decrease_step does not decrease step if it is too small ", "[nld::step_updater]") {
    auto min = 0.05;
    nld::step_updater updater(nld::step_updater_parameters{ min, 0.1, 2.0, 0 });

    auto expected_step = updater.step();
    updater.increase_step_if_possible();

    REQUIRE(updater.decrease_step());
    REQUIRE(updater.decrease_step() == false);
    REQUIRE(updater.step() >= min);
}
