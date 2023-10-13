#pragma once 

template<typename L, typename R>
auto are_equal(const L& l, const R& r) {
    if (l.size() != r.size())
        return false;

    for(auto i = 0u; i < l.size(); i++)
        if (l[i] != Approx(r[i]))
            return false;
    
    return true;
}
