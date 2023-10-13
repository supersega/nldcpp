#pragma once
#include <Eigen/Dense>

namespace nld::utils {

/// @brief The for_each algorithm for tuple.
/// @param t tuple for which we want apply function.
/// @param f function which we want to apply.
template <typename T, typename F>
void for_each(T&& t, F&& f) {
    std::apply([&f](auto&&... args) { (f(std::forward<decltype(args)>(args)), ...); }, std::forward<T>(t));
}

/// @brief count elements in tuple.
/// @param t which count we want to know.
/// @return count of 'joined' tuple elements.
template<typename T>
auto count(T&& t) -> std::size_t
{
    std::size_t n = 0;
    for_each(t, [&n](auto&& e) { n += e.size(); });
    return n;
}

/// @brief transform joined vector to tuple.
/// @param v vector which we want transform.
/// @param t tuple which we want.
template<typename V, typename T>
auto vector_to_tuple(const V& v, T& t) -> void {
    Eigen::Index n = 0;

    for_each(t, [&](auto&& e) {
        for(Eigen::Index i = 0; i < e.size(); i++)
            e[i] = v(i + n);

        n += e.size();
    });
}

/// @brief transform to tuple joined vector.
/// @param t tuple which we want to transform.
/// @param v vector which we want to get.
template<typename T, typename V>
auto tuple_to_vector(const T& t, V&& v) -> void {
    auto count = nld::utils::count(t);
    
    v.resize(count);

    Eigen::Index n = 0;
    for_each(t, [&](auto&& e) {
        for(Eigen::Index i = 0; i < e.size(); i++)
            v(i + n) = static_cast<double>(e[i]);

        n += e.size();
    });
}

namespace detail {
/// Create index sequence at interval [Start, End)
template <std::size_t Start, std::size_t End, std::size_t... Is>
auto make_index_sequence_impl() {
    if constexpr (End == Start) return std::index_sequence<Is...>();
    else return make_index_sequence_impl<Start, End - 1, End - 1, Is...>();
}

/// Create index sequence on [Start, End)
template <std::size_t Start, std::size_t End>
using make_index_sequence = std::decay_t<decltype(make_index_sequence_impl<Start, End>())>;

/// Create tuple view using index sequence
template <class Tuple, size_t... Is>
constexpr auto view_impl(Tuple&& t,
    std::index_sequence<Is...>) {
    return std::forward_as_tuple(std::get<Is>(t)...);
}
}

/// @brief Create view on tuple head size N.
/// @tparam N head size.
/// @param t tuple.
template <size_t N, class T>
constexpr auto head_view(T&& t) {
    constexpr auto size = std::tuple_size<std::remove_reference_t<T>>::value;
    static_assert(N <= size, "N must be smaller or equal than size of tuple");
    return detail::view_impl(t, detail::make_index_sequence<0, N>{});
}

/// @brief Create view on tuple last N elements.
/// @param t tuple.
template <size_t N, class T>
constexpr auto tail_view(T&& t) {
    constexpr auto size = std::tuple_size<std::remove_reference_t<T>>::value;
    static_assert(N <= size, "N must be smaller or equal than size of tuple");
    return detail::view_impl(t, detail::make_index_sequence<size - N, size>{});
}
}