#include <python/wrappers/arc_length.hpp>
#include <python/wrappers/types.hpp>

#include <boost/hana.hpp>
#include <pybind11/eigen.h>

using namespace boost::hana::literals;

namespace wrappers {

/// @brief Wrap the generator as an iterable object
template <typename T>
void export_generator_as_iterable(py::module &m, std::string_view name) {
    py::class_<cppcoro::generator<T>>(m, name.data())
        .def(py::init<>())
        .def(
            "__iter__",
            [](cppcoro::generator<T> &g) {
                return py::make_iterator(g.begin(), g.end());
            },
            py::keep_alive<0, 1>());
}

// TODO:
template <typename System, typename VectorX, typename Map>
auto arc_length(System system, nld::continuation_parameters cp, VectorX us,
                VectorX ts, Map map) {
    // How can I teransform the wrapped map to the nld mapper?
    return nld::arc_length(system, cp, us, ts, map);
}

auto generate_sequence() -> cppcoro::generator<std::tuple<int, int>> {
    for (int i = 0; i < 10; ++i) {
        co_yield std::make_tuple(i, i + 1);
    }
}

void wrap_generator(py::module &m) {
    auto prm = boost::hana::permutations(wrappers::periodic_mapper_three_tags);

    boost::hana::for_each(prm, [&m](auto p) {
        auto mappers_tuple =
            boost::hana::transform(p, [](auto x) { return x[0_c]; });

        auto names_tuple =
            boost::hana::transform(p, [](auto x) { return x[1_c]; });

        using concat = decltype(boost::hana::unpack(
            mappers_tuple,
            boost::hana::template_<wrappers::concat_mappers>))::type;

        std::string mapper_full_name = boost::hana::fold_left(
            names_tuple, std::string{},
            [](std::string acc, auto x) { return acc + x; });

        export_generator_as_iterable<typename concat::result>(
            m, "ArcLengthGenerator" + mapper_full_name);
    });
}

auto make_duffing() -> wrappers::non_autonomous::RnPlusOneToRnMapFnDual;

void wrap_arc_length_non_autonomous(py::module &m) {
    auto fs = wrappers::non_autonomous::functions;
    auto solvers = wrappers::solvers;
    auto prm = boost::hana::permutations(wrappers::periodic_mapper_three_tags);

    auto types = boost::hana::cartesian_product(
        boost::hana::make_tuple(fs, solvers, prm));

    boost::hana::for_each(
        types, boost::hana::fuse([&m](auto fs, auto solvers, auto prm) {
            using function_t = typename decltype(+fs[0_c])::type;
            using solver_t = typename decltype(+solvers[0_c])::type;
            using periodic_t =
                nld::internal::periodic<solver_t,
                                        nld::non_autonomous<function_t>>;

            auto mappers_tuple =
                boost::hana::transform(prm, [](auto x) { return x[0_c]; });
            auto mappers_name =
                boost::hana::transform(prm, [](auto x) { return x[1_c]; });
            using concat = decltype(boost::hana::unpack(
                mappers_tuple,
                boost::hana::template_<wrappers::concat_mappers>))::type;

            std::string function_name = fs[1_c];
            std::string solver_name = solvers[1_c];

            std::string mapper_full_name = boost::hana::fold_left(
                mappers_name, std::string{},
                [](std::string acc, auto x) { return acc + x; });

            m.def("arc_length", [](periodic_t fn, nld::vector_xd us_,
                                   concat map) {
                nld::vector_xdd us(us_.size());
                us = us_;
                // us << 0.0, 0.0, 0.37;
                nld::vector_xdd ts(3);
                ts << 0.0, 0.0, 1.0;
                nld::continuation_parameters params(
                    nld::newton_parameters(25, 0.000005), 10.1, 0.01, 0.01,
                    nld::direction::forward);
                auto ip = nld::periodic_parameters_constant{1, 200};
                auto bvp = nld::periodic<nld::runge_kutta_4>(
                    nld::non_autonomous(wrappers::make_duffing()), ip);
                std::cout << "Arc length 222" << std::endl;
                try {
                    return nld::arc_length(std::move(fn), params, us, ts, map);
                } catch (const std::exception &e) {
                    std::cout << "exept: " << e.what() << std::endl;
                    throw;
                }
            });
            // static_assert(std::is_same_v<int, first_mapper_t>);
        }));

    m.def("print_vector", [](nld::vector_xd d) {
        std::cout << d << std::endl;
        // std::cout << v << std::endl;
    });
}

void wrap_arc_length(py::module &m) {
    export_generator_as_iterable<std::tuple<int, int>>(m, "IntGenerator");
    m.def("generate_sequence", &generate_sequence);
    wrap_arc_length_non_autonomous(m);
    wrap_generator(m);
}
} // namespace wrappers
