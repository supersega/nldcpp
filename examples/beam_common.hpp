#pragma once

#include <nld/calculus.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>

namespace nld::mechanics {

/// @brief Simple representation of beam.
struct geometry final {
    auto area() const -> double { return height * width; }

    auto inertia() const -> double {
        auto d = height / 2.0;
        auto b = width / 2.0;
        auto I = 4.0 / 3.0 * b * d * d * d;
        return I;
    }

    double length; ///< Length of beam.
    double height; ///< Height of beam.
    double width;  ///< Width of beam.
};

/// @brief Simple representation of material.
struct material {
    double young_modulus; ///< Young's modulus.
    double density;       ///< Density.
};

/// @brief Simple representation of damage.
struct damage {
    double position; ///< Position of damage.
    double depth;    ///< Depth of damage.
};

/// @brief Simple representation of force.
struct force {
    double amplitude; ///< Amplitude of force.
};

struct beam final {
    geometry geometry; ///< Geometry of beam.
    material material; ///< Material of beam.
};

struct hinged_beam_bc : nld::boundary_condition {
    explicit hinged_beam_bc(double l) : length(l) {}

    auto value() const {
        return [l = length](auto x) -> nld::adnum { return x * (x - l); };
    }

private:
    double length; ///< Length of beam.
};
} // namespace nld::mechanics
