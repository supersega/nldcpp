#pragma once 

namespace nld {

/// @brief Base class for problem formulation.
template<typename Fn>
struct problem {
    /// @brief Create problem from function and dimension.
    /// @param f function.
    explicit problem(Fn&& f) : function(std::forward<Fn>(f)) { }

    /// @brief Get underlying callable function.
    auto underlying_function() const -> const Fn& {
        return function;
    }
protected:
    Fn function; ///< Underlying callable function.
};

template<typename F>
problem(F&&, std::size_t)->problem<F>;

}