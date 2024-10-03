constexpr auto PI = 3.14159265358979323846264338327950288;
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <matplotlibcpp.h>
#include <memory>
#include <ostream>
#include <queue>
#include <span>
#include <sys/wait.h>
#include <vector>

#include <nld/autocont.hpp>
#include <nld/core.hpp>
#include <nld/math.hpp>

#include <matplot/matplot.h>

namespace plt = matplotlibcpp;

struct Material {
    double density;
    double young_module;
};

struct Geometry {
    auto area() const -> double { return height * width; }
    auto inertia() const -> double {
        auto b = width / 2.0;
        auto h = height / 2.0;
        auto I = 4.0 / 3.0 * b * h * h * h;
        return I;
    }

    double length;
    double height;
    double width;
};

struct Crack {
    double position;
    double depth;
};

struct Beam {
    Material material;
    Geometry geometry;
};

struct Interval {
    double begin;
    double end;
};

auto S(auto alpha, double xi, double xi0) -> decltype(alpha) {
    return sin(alpha * (xi - xi0)) + sinh(alpha * (xi - xi0));
}

auto dS(auto alpha, double xi, double xi0) -> decltype(alpha) {
    return (alpha) * (cos(alpha * (xi - xi0)) + cosh(alpha * (xi - xi0)));
}

auto ddS(auto alpha, double xi, double xi0) -> decltype(alpha) {
    return (alpha * alpha) *
           (-sin(alpha * (1 - xi0)) + sinh(alpha * (1 - xi0)));
}

double heaviside(double xi, double xi0) {
    if (xi < xi0) {
        return 0.0;
    } else {
        return 1.0;
    }
}

struct CrackedBeam {
    Beam beam;
    std::vector<Crack> cracks;

private:
    std::vector<double> __bettai() const {
        double h = beam.geometry.height;
        std::vector<double> bettaarray;
        for (const auto &crack : cracks) {
            bettaarray.push_back(crack.depth / h);
        }
        return bettaarray;
    }

public:
    std::vector<double> lambdai() const {
        std::vector<double> bi = __bettai();

        auto C = [](double betta) -> double {
            return 5.346 * (1.86 * pow(betta, 2) - 3.95 * pow(betta, 3) +
                            16.375 * pow(betta, 4) - 37.226 * pow(betta, 5) +
                            76.81 * pow(betta, 6) - 126.9 * pow(betta, 7) +
                            172.0 * pow(betta, 8) - 143.97 * pow(betta, 9) +
                            66.56 * pow(betta, 10));
        };

        auto C_0 = [](double betta) -> double {
            double a = betta * (2.0 - betta);
            double b = 0.9 * pow(betta - 1, 2);
            return a / b;
        };

        auto lambdaa = [&](double betta) {
            double h = beam.geometry.height;
            double L = beam.geometry.length;
            return (h / L) * C(betta);
        };

        std::vector<double> li;
        for (const auto &betta : bi) {
            double lambda = lambdaa(betta);
            li.push_back(lambda);
        }
        return li;
    }

    std::vector<std::function<nld::dual(nld::dual)>> mui() const {
        auto mu0 = [&](nld::dual alpha) -> nld::dual {
            double ksi0 = cracks[0].position / beam.geometry.length;
            return -(pow(alpha, 2)) * sin(alpha * ksi0);
        };

        std::vector<std::function<nld::dual(nld::dual)>> mui_local;
        mui_local.push_back(mu0);
        return mui_local;
    }

    std::vector<std::function<nld::dual(nld::dual)>> upsiloni() const {
        auto upsilon0 = [this](nld::dual alpha) -> nld::dual {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return -(pow(alpha, 2)) * cos(alpha * ksi0);
        };
        std::vector<std::function<nld::dual(nld::dual)>> upsiloni_local = {
            upsilon0};
        return upsiloni_local;
    }

    std::vector<std::function<nld::dual(nld::dual)>> zetai() const {
        auto zeta0 = [this](nld::dual alpha) -> nld::dual {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return pow(alpha, 2) * sinh(alpha * ksi0);
        };
        std::vector<std::function<nld::dual(nld::dual)>> zetai_local = {zeta0};
        return zetai_local;
    }

    std::vector<std::function<nld::dual(nld::dual)>> etai() const {
        auto eta0 = [this](nld::dual alpha) -> nld::dual {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return pow(alpha, 2) * cosh(alpha * ksi0);
        };
        std::vector<std::function<nld::dual(nld::dual)>> etai_local = {eta0};
        return etai_local;
    }

    std::vector<double> cracks_positions_dimless() const {
        std::vector<double> poss;
        for (auto crack : this->cracks) {
            poss.push_back(crack.position);
        }
        for (int i = 0; i < poss.size(); i++) {
            poss[i] /= this->beam.geometry.length;
        }
        return poss;
    }

    auto A(auto alpha, double xi) const -> decltype(alpha) {
        std::vector<double> lambdai = this->lambdai();
        int n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto mu = static_cast<decltype(alpha)>(mui[i](alpha));
            sum += (lambdai[i] * mu * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + sin(alpha * xi);
    }

    auto B(auto alpha, double xi) const -> decltype(alpha) {
        std::vector<double> lambdai = this->lambdai();
        int n = this->cracks.size();
        auto upsiloni = this->upsiloni();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto upsilon = static_cast<decltype(alpha)>(upsiloni[i](alpha));
            sum += (lambdai[i] * upsilon * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + cos(alpha * xi);
    }

    auto C(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        int n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto zeta = static_cast<decltype(alpha)>(zetai[i](alpha));
            sum += (lambdai[i] * zeta * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + sinh(alpha * xi);
    }

    auto D(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        int n = this->cracks.size();
        auto etai = this->etai();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto eta = static_cast<decltype(alpha)>(etai[i](alpha));
            sum += (lambdai[i] * eta * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + cosh(alpha * xi);
    }

    auto dA(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto mu = static_cast<decltype(alpha)>(mui[i](alpha));
            sum += lambdai[i] * mu * dS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha)*cos(alpha * xi);
    }

    auto dC(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto zeta = static_cast<decltype(alpha)>(zetai[i](alpha));
            sum += lambdai[i] * zeta * dS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha)*cosh(alpha * xi);
    }

    auto ddA(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto mu = static_cast<decltype(alpha)>(mui[i](alpha));
            sum += lambdai[i] * mu * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum - (alpha * alpha) * sin(alpha * xi);
    }

    auto ddB(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto upsiloni = this->upsiloni();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto upsilon = static_cast<decltype(alpha)>(upsiloni[i](alpha));
            sum += lambdai[i] * upsilon * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum - (alpha * alpha) * cos(alpha);
    }

    auto ddC(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto zeta = static_cast<decltype(alpha)>(zetai[i](alpha));
            sum += lambdai[i] * zeta * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha * alpha) * sinh(alpha);
    }

    auto ddD(auto alpha, double xi) const -> decltype(alpha) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto etai = this->etai();
        auto xii = this->cracks_positions_dimless();
        decltype(alpha) sum = 0.0;
        for (int i = 0; i < n; i++) {
            auto eta = static_cast<decltype(alpha)>(etai[i](alpha));
            sum += lambdai[i] * eta * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha * alpha) * cosh(alpha);
    }
};

nld::dual chnc(nld::dual alpha) {
    return cos(alpha) * sinh(alpha) - sin(alpha) * cosh(alpha);
}

auto make_hcc_bf(CrackedBeam cracked_beam) {
    auto chc = [=](auto alpha) {
        auto a11 = cracked_beam.A(alpha, 1);
        auto a12 = cracked_beam.C(alpha, 1);
        auto a21 = cracked_beam.dA(alpha, 1);
        auto a22 = cracked_beam.dC(alpha, 1);
        return a11 * a22 - a12 * a21;
    };
    return chc;
}

auto make_chc_bf(CrackedBeam cracked_beam) {
    auto chc = [=](auto alpha) {
        auto a11 = cracked_beam.A(alpha, 1) - cracked_beam.C(alpha, 1);
        auto a12 = cracked_beam.B(alpha, 1) - cracked_beam.D(alpha, 1);
        auto a21 = cracked_beam.ddA(alpha, 1) - cracked_beam.ddC(alpha, 1);
        auto a22 = cracked_beam.ddB(alpha, 1) - cracked_beam.ddD(alpha, 1);
        return a11 * a22 - a12 * a21;
    };
    return chc;
}

// TODO: make it not so dummy, lazy ass
auto find_roots(auto fn, Interval interval, size_t discretization)
    -> Eigen::VectorXd {
    auto a = interval.begin;
    auto b = interval.end;
    auto len = b - a;
    auto dx = len / discretization;

    std::vector<double> roots;
    for (size_t i = 1; i < discretization + 1; i++) {
        auto x_prev = a + dx * (i - 1);
        auto x_curr = a + dx * (i);
        auto fn_prev = fn(x_prev);
        auto fn_curr = fn(x_curr);

        if (fn_prev * fn_curr < 0)
            roots.push_back(x_curr);
    }

    Eigen::Map<Eigen::VectorXd> roots_(roots.data(), roots.size());
    return roots_;
}

auto find_roots_newton(auto fn, Interval interval) {
    auto initial = find_roots(fn, interval, 100);

    for (std::size_t i = 0; i < initial.size(); i++) {
        nld::vector_xdd first_guess(1);
        first_guess(0) = initial[i];
        nld::newton_parameters np{10, 1.0e-6};
        auto wrp = [fn](const nld::vector_xdd &x) -> nld::vector_xdd {
            nld::vector_xdd y(1);
            y(0) = fn(x(0));
            return y;
        };
        if (nld::newton(wrp, nld::wrt(first_guess), nld::at(first_guess), np))
            initial[i] = static_cast<double>(first_guess(0));
    }

    return initial;
}

auto normalized_mode(auto fn) {
    double max = fn(0.0);
    double min = fn(0.0);

    size_t N = 100;
    for (size_t i = 1; i < N + 1; i++) {
        auto xi = (double)i / (double)(N);
        auto value = fn(xi);
        if (value > max)
            max = value;
        else if (value < min)
            min = value;
    }

    auto abs_max = std::max(max, std::abs(min));
    abs_max = 1.0;

    return [fn, abs_max](auto xi) { return fn(xi) / abs_max; };
}

auto make_eigen_mode_ch(CrackedBeam cracked_beam, int index) {
    auto eigen_frequences_equation = make_chc_bf(cracked_beam);
    auto alphai =
        find_roots(eigen_frequences_equation, Interval{3.7, 11.0}, 10000);
    auto alpha = alphai[index];

    auto em = [=](auto xi) {
        auto aux = cracked_beam.A(alpha, xi);
        auto coef = (cracked_beam.A(alpha, 1) - cracked_beam.C(alpha, 1)) /
                    (cracked_beam.B(alpha, 1) - cracked_beam.D(alpha, 1));
        aux -= coef * cracked_beam.B(alpha, xi);
        aux -= cracked_beam.C(alpha, xi);
        aux += coef * cracked_beam.D(alpha, xi);
        return aux;
    };

    auto normalized = normalized_mode(em);

    return normalized;
}

auto make_eigen_mode_hc(CrackedBeam cracked_beam, int index) {
    auto eigen_frequences_equation = make_hcc_bf(cracked_beam);
    auto alphai =
        find_roots(eigen_frequences_equation, Interval{3.7, 11.0}, 10000);
    auto alpha = alphai[index];

    auto em = [=](double xi) {
        auto aux = cracked_beam.A(alpha, xi);
        auto coef = -cracked_beam.A(alpha, 1) / cracked_beam.C(alpha, 1);
        aux += coef * cracked_beam.C(alpha, xi);
        return aux;
    };
    return em;
}

enum BC { CH };

template <typename BC>
struct BeamTraits;

struct CHTag {};

template <typename T>
decltype(auto) _S(double k, T xi) {
    return 0.5 * (cosh(k * xi) + cos(k * xi));
}

template <typename T>
decltype(auto) _T(double k, T xi) {
    return 0.5 * (sinh(k * xi) + sin(k * xi));
}

template <typename T>
decltype(auto) _U(double k, T xi) {
    return 0.5 * (cosh(k * xi) - cos(k * xi));
}

template <typename T>
decltype(auto) _V(double k, T xi) {
    return 0.5 * (sinh(k * xi) - sin(k * xi));
}

template <>
struct BeamTraits<CHTag> {
    BeamTraits(Beam beam) : beam{beam} { calculate_roots(); }

    auto Phi(std::size_t m) const {
        auto k = roots[m];
        auto w = 2.0;

        return [&, k, w](auto xi) {
            return w * (_U(k, xi) - (_S(k, 1.0) / _T(k, 1.0)) * _V(k, xi));
        };
    }

    auto dPhi(std::size_t m) const {
        auto k = roots[m];
        auto w = 2.0;

        return [&, k, w](auto xi) {
            return k * w * (_T(k, xi) - (_S(k, 1.0) / _T(k, 1.0)) * _U(k, xi));
        };
    }

    auto ddPhi(std::size_t m) const {
        auto k = roots[m];
        auto w = 2.0;

        return [&, k, w](auto xi) {
            return k * k * w *
                   (_S(k, xi) - (_S(k, 1.0) / _T(k, 1.0)) * _T(k, xi));
        };
    }

    auto frequencies() const -> nld::vector_xd {
        auto E = beam.material.young_module;
        auto rho = beam.material.density;
        auto A = beam.geometry.area();
        auto I = beam.geometry.inertia();

        auto L = beam.geometry.length;
        auto squared = roots.array().square();
        return squared / L / L * sqrt(E * I / rho / A);
    }

private:
    void calculate_roots() {
        roots = find_roots_newton(chnc, Interval{3.7, 11.0});
    }

    Beam beam;
    nld::vector_xd roots;
};

template <typename BC>
struct CrackedBeamTraits;

template <>
struct CrackedBeamTraits<CHTag> {

    CrackedBeamTraits(CrackedBeam cb) : cracked_beam(cb) {
        calculate_roots();
        calculate_lambdai();
        calculate_xi();
        calculate_coefficients();
        calculate_Psi();
        calculate_S_factor();
        calculate_normalization_factors();
    }

    ~CrackedBeamTraits() {
        std::cout << "CrackedBeamTraits destructor called" << std::endl;
    }

    auto S_overline(std::size_t m, std::size_t i) const {
        auto w = normalization_factors(m);
        return [&, w, m, i](double xi) { return w * S_overline_raw(m, i, xi); };
    }

    auto dS_overline(std::size_t m, std::size_t i) const {
        auto w = normalization_factors(m);
        return
            [&, w, m, i](double xi) { return w * dS_overline_raw(m, i, xi); };
    }

    auto betta(std::size_t m) const {
        auto w = normalization_factors(m);
        return [&, w, m](double xi) { return w * betta_raw(m, xi); };
    }

    auto dbetta(std::size_t m) const {
        auto w = normalization_factors(m);
        return [&, w, m](double xi) { return w * dbetta_raw(m, xi); };
    }

    /// @brief eigen mode function factory
    /// @param m mode number starting from zero
    auto Phi(std::size_t m) const {
        auto factor = normalization_factors(m);
        return [&, factor, m](auto xi) { return factor * mode(m, xi); };
    }

    auto frequencies() const -> nld::vector_xd {
        auto E = cracked_beam.beam.material.young_module;
        auto rho = cracked_beam.beam.material.density;
        auto A = cracked_beam.beam.geometry.area();
        auto I = cracked_beam.beam.geometry.inertia();

        auto L = cracked_beam.beam.geometry.length;
        auto squared = roots.array().square();
        return squared / L / L * sqrt(E * I / rho / A);
    }

private:
    double S_overline_raw(std::size_t m, std::size_t i, double xi) const {
        auto alpha = roots[m];
        return S_factor(m, i) * S(alpha, xi, xii(i));
    }

    double dS_overline_raw(std::size_t m, std::size_t i, double xi) const {
        auto alpha = roots[m];
        return S_factor(m, i) * dS(alpha, xi, xii(i));
    }

    double betta_raw(std::size_t m, double xi) const {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        return sin(alpha * xi) - Psi * cos(alpha * xi) - sinh(alpha * xi) +
               Psi * cosh(alpha * xi);
    }

    double dbetta_raw(std::size_t m, double xi) const {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        return alpha * (cos(alpha * xi) + Psi * sin(alpha * xi) -
                        cosh(alpha * xi) + Psi * sinh(alpha * xi));
    }

    auto mode(std::size_t m, auto xi) const {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        auto n = cracked_beam.cracks.size();

        auto sum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            sum += S_overline_raw(m, i, xi) * heaviside(xi, xii(i));
        }

        sum = sum + betta_raw(m, xi);

        return sum;
    }

    void calculate_roots() {
        auto eq = make_chc_bf(cracked_beam);
        roots = find_roots_newton(eq, Interval{3.7, 11.0});
    }

    void calculate_lambdai() {
        nld::vector_xd li(cracked_beam.cracks.size());
        auto liv = cracked_beam.lambdai();
        for (std::size_t i = 0; i < liv.size(); i++) {
            li(i) = liv[i];
        }
        lambdai = li;
    }

    void calculate_xi() {
        nld::vector_xd xiil(cracked_beam.cracks.size());
        auto xiiv = cracked_beam.cracks_positions_dimless();
        for (std::size_t i = 0; i < xiil.size(); i++) {
            xiil(i) = xiiv[i];
        }
        xii = xiil;
    }

    auto apply_coefficient_function(const auto &functions) {
        nld::matrix_xd values(roots.size(), functions.size());

        for (size_t row = 0; row < values.rows(); row++) {
            for (size_t col = 0; col < values.cols(); col++) {
                values(row, col) =
                    static_cast<double>(functions[col](roots[row]));
            }
        }

        return values;
    }

    void calculate_coefficients() {
        mui = apply_coefficient_function(cracked_beam.mui());
        upsiloni = apply_coefficient_function(cracked_beam.upsiloni());
        zetai = apply_coefficient_function(cracked_beam.zetai());
        etai = apply_coefficient_function(cracked_beam.etai());
    }

    void calculate_Psi() {
        nld::vector_xd psii(roots.size());
        for (size_t i = 0; i < roots.size(); i++) {
            auto alpha = roots[i];
            psii(i) = (cracked_beam.A(alpha, 1) - cracked_beam.C(alpha, 1)) /
                      (cracked_beam.B(alpha, 1) - cracked_beam.D(alpha, 1));
        }

        Psii = psii;
    }

    void calculate_S_factor() {
        nld::matrix_xd s_factor(roots.size(), cracked_beam.cracks.size());
        for (size_t row = 0; row < s_factor.rows(); row++) {
            for (size_t col = 0; col < s_factor.cols(); col++) {
                auto aux = mui(row, col) - Psii(row) * upsiloni(row, col) -
                           zetai(row, col) + Psii(row) * etai(row, col);
                aux *= lambdai(col);
                auto alpha = roots[row];
                s_factor(row, col) = 0.5 / alpha * aux;
            }
        }
        S_factor = s_factor;
    }

    void calculate_normalization_factors() {
        nld::vector_xd factors(roots.size());
        for (size_t i = 0; i < factors.size(); i++) {
            auto mode_raw = [&, i](auto xi) { return mode(i, xi); };
            auto distance = nld::integrate<nld::gauss_kronrod21>(
                [&mode_raw](auto xi) { return mode_raw(xi) * mode_raw(xi); },
                nld::segment{0.0, 1.0},
                nld::gauss_kronrod21::integration_options{1.0e-6, 1.0e-6});
            factors(i) = 1.0 / sqrt(distance);
        }
        normalization_factors = factors;
    }

private:
    CrackedBeam cracked_beam;

    nld::vector_xd roots;

    nld::vector_xd lambdai;
    nld::vector_xd xii;

    nld::matrix_xd mui;
    nld::matrix_xd upsiloni;
    nld::matrix_xd zetai;
    nld::matrix_xd etai;

    nld::vector_xd Psii;

    nld::matrix_xd S_factor;

    nld::vector_xd normalization_factors;
};

struct Force {
    double position;
    double amplitude;
};

#include <cassert>

template <typename BC>
struct CrackedCaddemiBeamDynamicSystem final {
    CrackedCaddemiBeamDynamicSystem(CrackedBeam cracked_beam, Force force,
                                    double friction,
                                    std::size_t degrees_of_freedom)
        : cracked_beam{cracked_beam}, force{force}, friction{friction},
          degrees_of_freedom{degrees_of_freedom},
          beam_traits{cracked_beam.beam}, cracked_beam_traits{cracked_beam} {
        calculate();
    }

    ~CrackedCaddemiBeamDynamicSystem() {
        assert(false && "CrackedCaddemiBeamDynamicSystem destructor called");
    }

    void test() {
        std::cout << "Mass crack\n"
                  << beam_traits.frequencies()(0) << std::endl;
    }

    auto operator()(const nld::vector_xdd &y) const -> nld::vector_xdd {
        if (switch_function(y))
            return opened_crack(y, 0.0, 0.0);
        else
            return closed_crack(y, 0.0, 0.0);
    }

    using switch_fn = std::function<bool(const nld::vector_xdd &)>;

private:
    auto closed_crack(const nld::vector_xdd &y, nld::dual t,
                      nld::dual omega) const {
        nld::vector_xdd dy(2 * degrees_of_freedom);
        nld::vector_xdd yc = y.head(degrees_of_freedom);
        auto yct = nld::utils::tensor_view(yc);

        auto f =
            stiffnes_no_crack.cwiseProduct(stiffnes_no_crack).cwiseProduct(yc);

        Eigen::Tensor<nld::dual, 4> G =
            R_bar_no_crack.template cast<nld::dual>();
        Eigen::Tensor<nld::dual, 1> fnlt =
            G.contract(yct, std::array{Eigen::IndexPair(3, 0)})
                .contract(yct, std::array{Eigen::IndexPair(2, 0)})
                .contract(yct, std::array{Eigen::IndexPair(1, 0)});
        Eigen::Map<nld::vector_xdd> fnl(fnlt.data(), degrees_of_freedom);

        dy.head(degrees_of_freedom) = y.tail(degrees_of_freedom);
        dy.tail(degrees_of_freedom) =
            /*-friction * y.tail(degrees_of_freedom)*/ -f - fnl; /* -
             force_no_crack * nld::dual(sin(t * 2.0 * PI))*/

        // dy *= 2.0 * PI;
        // dy /= omega;

        return dy;
    }

    auto opened_crack(const nld::vector_xdd &y, nld::dual t,
                      nld::dual omega) const {
        nld::vector_xdd dy(2 * degrees_of_freedom);
        nld::vector_xdd yc = y.head(degrees_of_freedom);
        auto yct = nld::utils::tensor_view(yc);

        auto f = stiffnes_crack.cwiseProduct(stiffnes_crack).cwiseProduct(yc);

        Eigen::Tensor<nld::dual, 4> G = R_bar_crack.template cast<nld::dual>();
        Eigen::Tensor<nld::dual, 1> fnlt =
            G.contract(yct, std::array{Eigen::IndexPair(3, 0)})
                .contract(yct, std::array{Eigen::IndexPair(2, 0)})
                .contract(yct, std::array{Eigen::IndexPair(1, 0)});
        Eigen::Map<nld::vector_xdd> fnl(fnlt.data(), degrees_of_freedom);

        dy.head(degrees_of_freedom) = y.tail(degrees_of_freedom);
        dy.tail(degrees_of_freedom) =
            /*-friction * y.tail(degrees_of_freedom)*/ -f - fnl; /* -
             force_crack * nld::dual(sin(t * 2.0 * PI));*/

        // dy *= 2.0 * PI;
        // dy /= omega;

        return dy;
    }

    void calculate() {
        switch_function = make_switch_function();
        mass_no_crack = calculate_mass(beam_traits);
        mass_crack = calculate_mass(cracked_beam_traits);
        stiffnes_no_crack = calculate_stiffness_no_crack();
        stiffnes_crack = calculate_stiffness_crack();
        first_frequence_no_crack = calculate_fitst_frequence_no_crack();
        force_no_crack = calculate_force_no_crack();
        force_crack = calculate_force_crack();
        R_bar_no_crack = calculate_R_bar_no_crack();
        R_bar_crack = calculate_R_bar_crack();
    }

    auto make_switch_function() const -> switch_fn {
        auto xic0 = cracked_beam.cracks_positions_dimless()[0];
        Eigen::VectorXi space = Eigen::VectorXi::LinSpaced(
            degrees_of_freedom, 0, degrees_of_freedom - 1);
        nld::vector_xd curvature_at_crack = space.unaryExpr(
            [&](auto i) { return cracked_beam_traits.Phi(i)(xic0); });

        return [&, curvature_at_crack](const nld::vector_xdd &y) -> bool {
            nld::vector_xdd yc =
                y.head(degrees_of_freedom).template cast<double>();
            return curvature_at_crack.cwiseProduct(yc).sum() > 0.0;
        };
    }

    nld::matrix_xd calculate_mass(auto traits) const {
        std::size_t N = degrees_of_freedom;
        nld::matrix_xd mass(N, N);

        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                auto fn = [&traits, i, j](auto xi) -> double {
                    return traits.Phi(i)(xi) * traits.Phi(j)(xi);
                };
                mass(i, j) = nld::integrate<nld::gauss_kronrod21>(
                    fn, nld::segment{0.0, 1.0},
                    nld::gauss_kronrod21::integration_options{1.0e-6, 1.0e-6});
            }
        }

        auto rho = cracked_beam.beam.material.density;
        auto A = cracked_beam.beam.geometry.area();

        return rho * A * mass;
    }

    auto calculate_stiffness_no_crack() const -> nld::vector_xd {
        auto frequencies = beam_traits.frequencies();
        double Omega0 = beam_traits.frequencies()(0);
        frequencies /= Omega0;
        return frequencies;
    }

    auto calculate_stiffness_crack() const -> nld::vector_xd {
        auto frequencies = cracked_beam_traits.frequencies();
        double Omega0 = beam_traits.frequencies()(0);
        frequencies /= Omega0;
        return frequencies;
    }

    double calculate_fitst_frequence_no_crack() {
        return beam_traits.frequencies()(0);
    }

    auto calculate_force_no_crack() -> nld::vector_xd {
        auto N = degrees_of_freedom;
        nld::vector_xd F_(N);

        auto h = cracked_beam.beam.geometry.height / 2.0;
        auto xif = force.position / cracked_beam.beam.geometry.length;
        auto F = force.amplitude;
        auto Omega0 = beam_traits.frequencies()(0);

        for (size_t i = 0; i < N; i++) {
            F_(i) = F * beam_traits.Phi(i)(xif) / Omega0 / Omega0 / h /
                    sqrt(cracked_beam.beam.geometry.length);
        }

        auto Fnc = mass_no_crack.inverse() * F_;

        return Fnc;
    }

    auto calculate_force_crack() -> nld::vector_xd {
        auto N = degrees_of_freedom;
        nld::vector_xd F_(N);

        auto h = cracked_beam.beam.geometry.height / 2.0;
        auto xif = force.position / cracked_beam.beam.geometry.length;
        auto F = force.amplitude;
        auto Omega0 = beam_traits.frequencies()(0);

        for (size_t i = 0; i < N; i++) {
            F_(i) = F * cracked_beam_traits.Phi(i)(xif) / Omega0 / Omega0 / h /
                    sqrt(cracked_beam.beam.geometry.length);
        }

        auto Fc = mass_crack.inverse() * F_;

        return Fc;
    }

    Eigen::Tensor<double, 4> calculate_R_bar_no_crack() {
        auto N = degrees_of_freedom;
        Eigen::Tensor<double, 4> R(N, N, N, N);
        auto E = cracked_beam.beam.material.young_module;
        auto A = cracked_beam.beam.geometry.area();
        auto L = cracked_beam.beam.geometry.length;
        auto h = cracked_beam.beam.geometry.height / 2.0;
        auto Omega0 = beam_traits.frequencies()(0);

        // TODO Symetry
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t k = 0; k < N; k++) {
                    for (size_t l = 0; l < N; l++) {
                        auto make_integrand = [&](auto ii, auto jj) {
                            return [&, ii, jj](auto xi) {
                                return beam_traits.dPhi(ii)(xi) *
                                       beam_traits.dPhi(jj)(xi);
                            };
                        };

                        auto R_ij = nld::integrate<nld::gauss_kronrod21>(
                            make_integrand(i, j), nld::segment{0.0, 1.0},
                            nld::gauss_kronrod21::integration_options{1.0e-6,
                                                                      1.0e-6});
                        auto R_kl = nld::integrate<nld::gauss_kronrod21>(
                            make_integrand(k, l), nld::segment{0.0, 1.0},
                            nld::gauss_kronrod21::integration_options{1.0e-6,
                                                                      1.0e-6});

                        auto R_bar_ijkl = (h * h / Omega0 / Omega0) * (E * A) /
                                          2.0 / pow(L, 5) * R_ij * R_kl;
                        R(i, j, k, l) = R_bar_ijkl;
                    }
                }
            }
        }

        std::array dimensions = {Eigen::IndexPair(0, 1)};
        nld::matrix_xd mass_no_crack_inversed = mass_no_crack.inverse();
        Eigen::Tensor<double, 4> R_ = R.contract(
            nld::utils::tensor_view(mass_no_crack_inversed), dimensions);

        return R_;
    }

    Eigen::Tensor<double, 4> calculate_R_bar_crack() {
        auto N = degrees_of_freedom;
        Eigen::Tensor<double, 4> R(N, N, N, N);
        auto E = cracked_beam.beam.material.young_module;
        auto A = cracked_beam.beam.geometry.area();
        auto L = cracked_beam.beam.geometry.length;
        auto h = cracked_beam.beam.geometry.height / 2.0;
        auto Omega0 = beam_traits.frequencies()(0);
        auto cracks_positions_dimless = cracked_beam.cracks_positions_dimless();

        auto calculate_R_nm = [&](auto n, auto m) -> double {
            auto N_cracks = cracked_beam.cracks.size();

            auto R_nm = 0.0;
            for (size_t i = 0; i < N_cracks; i++) {
                auto xii = cracks_positions_dimless[i];
                for (size_t j = 0; j < N_cracks; j++) {
                    auto S_ijnm = [&, i, j, n, m](auto xi) {
                        return cracked_beam_traits.dS_overline(n, i)(xi) *
                               cracked_beam_traits.dS_overline(m, j)(xi);
                    };

                    auto xij = cracks_positions_dimless[j];
                    auto R_ij =
                        1.0 * nld::integrate<nld::gauss_kronrod21>(
                                  S_ijnm, nld::segment{std::max(xii, xij), 1.0},
                                  nld::gauss_kronrod21::integration_options{
                                      1.0e-6, 1.0e-6});

                    auto a = [](auto ii, auto jj) -> double {
                        if (jj < ii)
                            return 0.0;
                        if (jj == ii)
                            return 0.5;
                        return 1.0;
                    };

                    R_ij += 1.0 * a(i, j) *
                            cracked_beam_traits.dS_overline(n, i)(xij) *
                            cracked_beam_traits.S_overline(m, j)(xij);

                    auto b = [](auto ii, auto jj) -> double {
                        if (ii < jj)
                            return 0.0;
                        if (jj == ii)
                            return 0.5;
                        return 1.0;
                    };

                    R_ij += 1.0 * b(i, j) *
                            cracked_beam_traits.S_overline(n, i)(xii) *
                            cracked_beam_traits.dS_overline(m, j)(xii);
                    R_nm += R_ij;
                }
                auto Adel = 2.013;
                R_nm += 1.0 * cracked_beam_traits.S_overline(n, i)(xii) *
                        cracked_beam_traits.S_overline(m, i)(xii);

                auto R1_aux = [&, i, n, m](auto xi) -> double {
                    return cracked_beam_traits.dS_overline(n, i)(xi) *
                           cracked_beam_traits.dbetta(m)(xi);
                };
                auto R2_aux = [&, i, n, m](auto xi) -> double {
                    return cracked_beam_traits.dS_overline(m, i)(xi) *
                           cracked_beam_traits.dbetta(n)(xi);
                };

                R_nm += 1.0 * nld::integrate<nld::gauss_kronrod21>(
                                  R1_aux, nld::segment{xii, 1.0},
                                  nld::gauss_kronrod21::integration_options{
                                      1.0e-6, 1.0e-6});
                R_nm += 1.0 * nld::integrate<nld::gauss_kronrod21>(
                                  R2_aux, nld::segment{xii, 1.0},
                                  nld::gauss_kronrod21::integration_options{
                                      1.0e-6, 1.0e-6});
                R_nm += 1.0 * nld::integrate<nld::gauss_kronrod21>(
                                  [&](auto xi) {
                                      return cracked_beam_traits.dbetta(n)(xi) *
                                             cracked_beam_traits.dbetta(m)(xi);
                                  },
                                  nld::segment{0, 1.0},
                                  nld::gauss_kronrod21::integration_options{
                                      1.0e-6, 1.0e-6});
            }
            return R_nm;
        };

        // TODO Symetry
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t k = 0; k < N; k++) {
                    for (size_t l = 0; l < N; l++) {
                        auto make_integrand = [&](auto ii, auto jj) {
                            return [&, ii, jj](auto xi) {
                                return beam_traits.dPhi(ii)(xi) *
                                       beam_traits.dPhi(jj)(xi);
                            };
                        };

                        auto R_ij = calculate_R_nm(i, j);
                        auto R_kl = calculate_R_nm(k, l);

                        auto R_bar_ijkl = (h * h / Omega0 / Omega0) * (E * A) /
                                          2.0 / pow(L, 5) * R_ij * R_kl;
                        R(i, j, k, l) = R_bar_ijkl;
                    }
                }
            }
        }

        std::array dimensions = {Eigen::IndexPair(0, 1)};
        nld::matrix_xd mass_crack_inversed = mass_crack.inverse();
        Eigen::Tensor<double, 4> R_ = R.contract(
            nld::utils::tensor_view(mass_crack_inversed), dimensions);

        return R_;
    }

    CrackedBeam cracked_beam;
    Force force;
    double friction;
    std::size_t degrees_of_freedom;
    BeamTraits<BC> beam_traits;
    CrackedBeamTraits<BC> cracked_beam_traits;

    switch_fn switch_function;
    nld::matrix_xd mass_no_crack;
    nld::matrix_xd mass_crack;
    nld::vector_xd stiffnes_no_crack;
    nld::vector_xd stiffnes_crack;
    nld::vector_xd force_no_crack;
    nld::vector_xd force_crack;
    double first_frequence_no_crack;
    Eigen::Tensor<double, 4> R_bar_no_crack;
    Eigen::Tensor<double, 4> R_bar_crack;

    std::unique_ptr<int> y0 = std::make_unique<int>(0);
};

void integrate_eigen_mode(CrackedBeam cracked_beam) {
    auto fem = make_eigen_mode_hc(cracked_beam, 0);
    auto sem = make_eigen_mode_hc(cracked_beam, 1);
    auto mass = [=](auto xi) mutable { return sem(xi) * fem(xi); };
    auto value = nld::integrate<nld::gauss_kronrod21>(
        mass, nld::segment{0.0, 1.0},
        nld::gauss_kronrod21::integration_options{1.0e-6, 1.0e-6});
}

auto make_eigen_mode_no_krack(size_t index) {
    std::array<double, 3> roots = {3.93, 7.07, 10.21};
    auto k = roots[index];
    auto _S = [k](auto xi) { return 0.5 * (cosh(k * xi) + cos(k * xi)); };
    auto _T = [k](auto xi) { return 0.5 * (sinh(k * xi) + sin(k * xi)); };
    auto _U = [k](auto xi) { return 0.5 * (cosh(k * xi) - cos(k * xi)); };
    auto _V = [k](auto xi) { return 0.5 * (sinh(k * xi) - sin(k * xi)); };

    return [=](auto xi) { return _U(xi) - (_S(1.0) / _T(1.0)) * _V(xi); };
}

auto make_eigen_mode_no_krack2(size_t index) {
    std::array<double, 3> roots = {3.93, 7.07, 10.21};
    auto k = roots[index];

    return [=](auto xi) {
        return sin(k * xi) - (sin(k) / sinh(k)) * sinh(k * xi);
    };
}

void integrate_eigen_mode2(CrackedBeam cracked_beam) {
    auto fem = make_eigen_mode_no_krack(0);
    auto sem = make_eigen_mode_no_krack(1);
    auto mass = [=](auto xi) mutable { return sem(xi) * fem(xi); };
    auto value = nld::integrate<nld::gauss_kronrod21>(
        mass, nld::segment{0.0, 1.0},
        nld::gauss_kronrod21::integration_options{1.0e-6, 1.0e-6});
}

void remove_element(auto &v, unsigned int index) {
    auto size = v.size() - 1;

    if (index < size)
        v.segment(index, size - index) = v.segment(index + 1, size - index);

    v.conservativeresize(size);
}

void remove_column(auto &matrix, unsigned int coltoremove) {
    auto numrows = matrix.rows();
    auto numcols = matrix.cols() - 1;

    if (coltoremove < numcols)
        matrix.block(0, coltoremove, numrows, numcols - coltoremove) =
            matrix.block(0, coltoremove + 1, numrows, numcols - coltoremove);

    matrix.conservativeresize(numrows, numcols);
}

template <typename Ds>
struct bind_wrt_unknown : nld::jacobian_mixin<bind_wrt_unknown<Ds>> {
    bind_wrt_unknown(Ds &ds, std::size_t coord, double value, std::size_t dim)
        : coord{coord}, value{value}, ds{ds}, dim{dim} {}

    /// @brief operator() to be used by The Newton method
    /// @param u the unknown vector [u_1, u2, ..., u_coord - 1, u_coord + 1,
    /// ..., u_dim + 1]
    auto operator()(const auto &u) const -> nld::vector_xdd {
        nld::vector_xdd u_merged = merged_unknown(u);
        return ds(u_merged);
    }

    auto merged_unknown(const auto &u) const -> nld::vector_xdd {
        nld::vector_xdd u_merged(dim + 1);
        u_merged << u.head(coord), value, u.tail(dim - coord);
        return u_merged;
    }

private:
    std::size_t coord;
    double value;
    std::size_t dim;
    Ds &ds;
};

auto compute_initial_estimation_free(const auto &cbvp, auto dofs)
    -> nld::vector_xdd {
    auto dim = 2 * dofs;
    nld::continuation_parameters params(nld::newton_parameters(100, 0.00005),
                                        1.0, 0.001, 0.01,
                                        nld::direction::reverse);

    auto ip =
        nld::periodic_parameters_adaptive{1, 1.0 / 2048, 1.0 / 128.0, 5.0e-6};

    auto ode = cbvp.underlying_function();
    auto bvp =
        nld::periodic<nld::runge_kutta_45>(cbvp.underlying_function(), ip);
    // auto bind = nld::bind_wrt_unknown(bvp, 0, 0.025, dim);
    auto bind = nld::bind_wrt_unknown(bvp, 0, 0.5, dim);

    nld::vector_xdd u0(dim);
    u0 << 0.00473749, -0.00433584, 0.0912594, -0.011132, -0.0523547, 4.48523;
    // std::cout << "Initial value is : " << u0 << '\n';
    // u0 << nld::vector_xdd::Zero(dim - 1), 5.01818;

    if (nld::newton(bind, wrt(u0), at(u0), nld::newton_parameters(50, 1.0e-6)))
        std::cout << "Root value is : " << u0 << '\n';
    else
        std::cout << "Newton method does not converged \n";

    return bind.merged_unknown(u0);
}

auto initial_guess(auto &ds, nld::collocations::mesh_parameters parameters,
                   std::size_t dofs) -> nld::vector_xdd {
    auto dimension = 2 * dofs;

    auto ig = compute_initial_estimation_free(ds, dofs);

    return nld::make_collocation_unknowns(ds, ig);
}

auto to_std_vector(const nld::vector_xdd &v) {
    std::vector<double> std_v(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        std_v[i] = static_cast<double>(v(i));
    }
    return std_v;
}

void plot_collection_of_curves(std::span<std::vector<double>> xss,
                               std::span<std::vector<double>> yss) {
    for (std::size_t i = 0; i < xss.size(); ++i) {
        auto xs = xss[i];
        auto ys = yss[i];
        plt::plot(xs, ys);
    }
}

void plot_curve_with_dots(std::span<double> xs, std::span<double> ys,
                          std::span<double> pxs, std::span<double> pys) {
    plt::plot(xs, ys, "k-");

    for (std::size_t i = 0; i < pxs.size(); ++i) {
        auto px = pxs[i];
        auto py = pys[i];
        plt::plot({px}, {py}, {{"marker", "o"}, {"linestyle", ""}});
    }
}

int main(int argc, char *argv[]) {
    auto material = Material{7800, 2.1e11};
    auto geometry = Geometry{0.177, 0.01, 0.01};
    auto crack = Crack{geometry.length * 0.5, geometry.height * 0.4};
    auto beam = Beam{material, geometry};
    auto cracked_beam = CrackedBeam{beam, std::vector<Crack>{crack}};

    auto fem = make_eigen_mode_ch(cracked_beam, 0);
    auto sem = make_eigen_mode_hc(cracked_beam, 0);
    auto mass = [=](auto xi) mutable { return sem(xi) * fem(xi); };
    CrackedBeamTraits<CHTag> chcbeam(cracked_beam);
    auto semnc = chcbeam.Phi(2);

    BeamTraits<CHTag> traits(beam);
    const size_t dofs = 3;
    CrackedCaddemiBeamDynamicSystem<CHTag> ds{
        cracked_beam, Force{beam.geometry.length / 2.0, 1'500.0}, 0.01, dofs};

    nld::collocations::mesh_parameters mesh_parameters{50, 4};
    nld::continuation_parameters params(nld::newton_parameters(100, 0.00001),
                                        19.5, 0.001, 0.01,
                                        nld::direction::reverse);

    auto basis_builder = nld::collocations::make_basis_builder<
        nld::collocations::lagrange_basis>();
    nld::periodic_collocations bvp(nld::autonomous(ds), basis_builder,
                                   mesh_parameters, dofs * 2);

    auto initial = initial_guess(bvp, mesh_parameters, dofs);
    nld::vector_xdd u0(initial.size());
    u0 << initial;

    nld::vector_xdd v0(initial.size());
    v0 << -1.0, nld::vector_xdd::Zero(initial.size() - 1);

    std::cout << "Start\n";

    std::queue<double> frequencies;
    frequencies.push(1.34);
    frequencies.push(1.4);
    frequencies.push(1.46);
    frequencies.push(1.51);
    // frequencies.push(1.4);
    // frequencies.push(1.43);
    // frequencies.push(1.46);
    // frequencies.push(1.49);
    // frequencies.push(1.51);
    // frequencies.push(1.52);
    // amplitudes.push(0.05);
    // amplitudes.push(0.1);
    // amplitudes.push(0.15);
    // amplitudes.push(0.2);
    // amplitudes.push(0.25);
    // amplitudes.push(0.275);
    // amplitudes.push(0.5);
    // amplitudes.push(0.525);
    // amplitudes.push(0.545);
    // amplitudes.push(0.525);
    // amplitudes.push(0.5);
    // amplitudes.push(0.45);
    // amplitudes.push(0.4);
    // amplitudes.push(0.375);
    std::vector<std::vector<double>> u0s;
    std::vector<std::vector<double>> u1s;
    std::vector<std::vector<double>> u2s;
    std::vector<std::vector<double>> tss;
    std::vector<double> Omega, A1;
    std::vector<double> Omega_selected, A1_selected;
    size_t counter = 0;
    size_t counter2 = 0;
    plt::figure_size(1200, 600);
    for (auto [solution, mean_amplitude, u0, u1, u2, ts] : nld::arc_length(
             nld::periodic_collocations(nld::autonomous(ds), basis_builder,
                                        mesh_parameters, dofs * 2),
             params, u0,
             concat(nld::solution(), nld::mean_amplitude(0),
                    nld::generalized_coordinate(0),
                    nld::generalized_coordinate(1),
                    nld::generalized_coordinate(2), nld::timestamps()))) {
        double om = (double)nld::dual(2.0 * PI / solution(solution.size() - 1));
        if (om < 0.0)
            break;

        auto A1_ = (double)(mean_amplitude);

        Omega.push_back((double)om);
        A1.push_back(A1_);

        // auto x = to_std_vector(u0);
        // auto y = to_std_vector(u1);

        // if (counter % 50 == 0) {
        //     plt::clf();
        //     plt::plot(x, y);
        //     plt::ylabel(R"($q_2$)");
        //     plt::xlabel(R"($q_1$)");
        //
        //     std::stringstream ss;
        //     ss << std::setw(4) << std::setfill('0') << counter2;
        //
        //     std::string filename = "anim/frame_" + ss.str() + ".png";
        //     std::cout << "Save frame: " << filename << std::endl;
        //     plt::save(filename);
        //     counter2++;
        // }
        // counter++;

        auto target_frequency = frequencies.front();
        if (std::abs(om - target_frequency) < 0.005) {
            Omega_selected.push_back((double)om);
            A1_selected.push_back(A1_);
            u0s.push_back(to_std_vector(u0));
            u1s.push_back(to_std_vector(u1));
            u2s.push_back(to_std_vector(u2));
            tss.push_back(to_std_vector(ts));
            std::cout << "Store NNM for Amplitude: " << om << "A1: " << A1_
                      << std::endl;
            std::cout << "u size: " << u0.size() << std::endl;
            std::cout << "t size: " << ts.size() << std::endl;
            std::cout << "timestamps:\n" << ts << std::endl;
            frequencies.pop();
        }

        std::cout << "Omega: " << om << "A1: " << (double)mean_amplitude
                  << std::endl;
    }

    // auto timestamps = tss.back();
    // auto _u0 = u0s.back();
    // auto _u1 = u1s.back();
    // auto _u2 = u2s.back();

    // for (std::size_t i = 0; i < _u0.size(); ++i) {
    //     auto u0i = _u0[i];
    //     auto u1i = _u1[i];
    //     auto u2i = _u2[i];
    //
    //     auto N = 200;
    //     auto dx = 1.0 / N;
    //
    //     std::vector<double> x(N + 1);
    //     std::vector<double> y(N + 1);
    //
    //     for (std::size_t i = 0; i < N + 1; ++i) {
    //         x[i] = i * dx;
    //         y[i] = traits.Phi(0)(x[i]) * u0i + traits.Phi(1)(x[i]) * u1i +
    //                traits.Phi(2)(x[i]) * u2i;
    //     }
    //
    //     plt::clf();
    //     plt::xlim(0.0, 1.0);
    //     plt::ylim(-1.0, 1.0);
    //     plt::plot(x, y);
    //     plt::ylabel(R"($\frac{y}{h}$)");
    //     plt::xlabel(R"($\frac{x}{l}$)");
    //     std::stringstream ss;
    //     ss << std::setw(4) << std::setfill('0') << i;
    //
    //     std::string filename = "anim/frame_" + ss.str() + ".png";
    //
    //     std::cout << "Save frame: " << filename << std::endl;
    //
    //     plt::save(filename);
    // }

    plt::figure_size(1200, 600);
    //
    plt::subplot(1, 2, 1);
    plot_curve_with_dots(Omega, A1, Omega_selected, A1_selected);
    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");

    plt::subplot(1, 2, 2);
    plot_collection_of_curves(u0s, u2s);
    plt::ylabel(R"($q_3$)");
    plt::xlabel(R"($q_1$)");
    //
    plt::save("internal_resonans_A3_q1q3.pdf");

    plt::show();

    return 0;
}
