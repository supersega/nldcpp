#pragma once

#include <nld/calculus/diff.hpp>

namespace nld {
template<typename E>
auto laplacian(E&& expression) {
    if constexpr (std::remove_reference_t<E>::dimension == 1) {
        auto& [x] = expression.get_space().coords();
        return nld::diff(std::forward<E>(expression), autodiff::wrt(x, x));
    }
}
}
