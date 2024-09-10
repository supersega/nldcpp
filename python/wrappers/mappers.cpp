#include <python/wrappers/mappers.hpp>

#include <boost/hana.hpp>

#include <tuple>

using namespace boost::hana::literals;

namespace wrappers {

template <typename... Args>
struct concat_maker {
    auto make() {
        return [](Args... args) { return wrappers::concat_mappers(args...); };
    }
};

template <typename... Args>
struct mapper_maker {

    template <typename Mapper>
    auto make() {
        return [](Args... args) { return Mapper(args...); };
    }
};

template <typename... Args>
auto to_tuple_t(std::tuple<Args...>) {
    return boost::hana::make_tuple(boost::hana::type_c<Args>...);
}

void wrap_mapper_functions(py::module &m) {
    boost::hana::for_each(wrappers::mapper_tags, [&m](auto x) {
        using mapper = typename decltype(+x[0_c])::type;
        using result = typename mapper::result;

        std::string name = x[1_c];
        std::string maker_function_name = x[2_c];

        py::class_<mapper>(m, name.c_str()).doc() = "Mapper";

        using hana_args =
            decltype(to_tuple_t(std::declval<typename mapper::args>()));
        //
        using mapper_maker_type = decltype(boost::hana::unpack(
            std::declval<hana_args>(),
            boost::hana::template_<wrappers::mapper_maker>))::type;

        m.def(maker_function_name.c_str(),
              mapper_maker_type{}.template make<mapper>());
    });
}

auto wrap_concat(py::module &m, auto &prm) {
    boost::hana::for_each(prm, [&m](auto p) {
        auto mappers_tuple =
            boost::hana::transform(p, [](auto x) { return x[0_c]; });

        auto names_tuple =
            boost::hana::transform(p, [](auto x) { return x[1_c]; });

        using concat = decltype(boost::hana::unpack(
            mappers_tuple,
            boost::hana::template_<wrappers::concat_mappers>))::type;

        using make_concat = decltype(boost::hana::unpack(
            mappers_tuple, boost::hana::template_<concat_maker>))::type;

        std::string name = boost::hana::fold_left(
            names_tuple, std::string{},
            [](std::string acc, auto x) { return acc + x; });

        auto mapper_name = name + "Mapper";
        auto result_name = name + "Result";

        py::class_<concat>(m, mapper_name.c_str()).doc() =
            "Concatenated mappers";

        m.def("concat", make_concat{}.make());
    });
}

void wrap_concat(py::module &m) {
    auto prm3 = boost::hana::permutations(wrappers::periodic_mapper_three_tags);
    auto prm2 = boost::hana::permutations(wrappers::periodic_mapper_two_tags);

    wrap_concat(m, prm3);
    wrap_concat(m, prm2);
}

void wrap_mappers(py::module &m) {
    wrap_mapper_functions(m);
    wrap_concat(m);
}
} // namespace wrappers
