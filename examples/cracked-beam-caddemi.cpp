#include "nld/core/aliases.hpp"
#include "nld/math/gauss_kronrod21.hpp"
#include "nld/math/integrate.hpp"
#include "nld/math/segment.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <matplotlibcpp.h>
#include <sys/wait.h>
#include <vector>

#include <nld/math.hpp>

namespace plt = matplotlibcpp;

struct Material {
    double density;
    double young_module;
};

struct Geometry {
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

double S(double alpha, double xi, double xi0) {
    return sin(alpha * (xi - xi0)) + sinh(alpha * (xi - xi0));
}

double dS(double alpha, double xi, double xi0) {
    return (alpha) * (cos(alpha * (xi - xi0)) + cosh(alpha * (xi - xi0)));
}

double ddS(double alpha, double xi, double xi0) {
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
    std::vector<double> __bettai() {
        double h = beam.geometry.height;
        std::vector<double> bettaarray;
        for (const auto &crack : cracks) {
            bettaarray.push_back(crack.depth / h);
        }
        return bettaarray;
    }

public:
    std::vector<double> lambdai() {
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
            return (h / L) * C_0(betta);
        };

        std::vector<double> li;
        for (const auto &betta : bi) {
            double lambda = lambdaa(betta);
            li.push_back(lambda);
        }
        return li;
    }

    std::vector<std::function<double(double)>> mui() {
        auto mu0 = [&](double alpha) {
            double ksi0 = cracks[0].position / beam.geometry.length;
            return -(pow(alpha, 2)) * sin(alpha * ksi0);
        };

        std::vector<std::function<double(double)>> mui_local;
        mui_local.push_back(mu0);
        return mui_local;
    }

    std::vector<std::function<double(double)>> upsiloni() {
        auto upsilon0 = [this](double alpha) {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return -(std::pow(alpha, 2)) * std::cos(alpha * ksi0);
        };
        std::vector<std::function<double(double)>> upsiloni_local = {upsilon0};
        return upsiloni_local;
    }

    std::vector<std::function<double(double)>> zetai() {
        auto zeta0 = [this](double alpha) {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return std::pow(alpha, 2) * std::sinh(alpha * ksi0);
        };
        std::vector<std::function<double(double)>> zetai_local = {zeta0};
        return zetai_local;
    }

    std::vector<std::function<double(double)>> etai() {
        auto eta0 = [this](double alpha) {
            double ksi0 = this->cracks[0].position / this->beam.geometry.length;
            return std::pow(alpha, 2) * std::cosh(alpha * ksi0);
        };
        std::vector<std::function<double(double)>> etai_local = {eta0};
        return etai_local;
    }

    std::vector<double> cracks_positions_dimless() {
        std::vector<double> poss;
        for (auto crack : this->cracks) {
            poss.push_back(crack.position);
        }
        for (int i = 0; i < poss.size(); i++) {
            poss[i] /= this->beam.geometry.length;
        }
        return poss;
    }

    double A(double alpha, double xi) {
        std::vector<double> lambdai = this->lambdai();
        int n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (lambdai[i] * mui[i](alpha) * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + std::sin(alpha * xi);
    }

    double B(double alpha, double xi) {
        std::vector<double> lambdai = this->lambdai();
        int n = this->cracks.size();
        auto upsiloni = this->upsiloni();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (lambdai[i] * upsiloni[i](alpha) * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + cos(alpha * xi);
    }

    double C(double alpha, double xi) {
        auto lambdai = this->lambdai();
        int n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (lambdai[i] * zetai[i](alpha) * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + sinh(alpha * xi);
    }

    double D(double alpha, double xi) {
        auto lambdai = this->lambdai();
        int n = this->cracks.size();
        auto etai = this->etai();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += (lambdai[i] * etai[i](alpha) * S(alpha, xi, xii[i]) *
                    heaviside(xi, xii[i]));
        }
        return (0.5 / alpha) * sum + cosh(alpha * xi);
    }

    double dA(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * mui[i](alpha) * dS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha)*cos(alpha * xi);
    }

    double dC(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * zetai[i](alpha) * dS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha)*cosh(alpha * xi);
    }

    double ddA(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto mui = this->mui();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * mui[i](alpha) * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum - (alpha * alpha) * sin(alpha * xi);
    }

    double ddB(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto upsiloni = this->upsiloni();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * upsiloni[i](alpha) * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum - (alpha * alpha) * cos(alpha);
    }

    double ddC(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto zetai = this->zetai();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * zetai[i](alpha) * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha * alpha) * sinh(alpha);
    }

    double ddD(double alpha, double xi) {
        auto lambdai = this->lambdai();
        auto n = this->cracks.size();
        auto etai = this->etai();
        auto xii = this->cracks_positions_dimless();
        auto sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += lambdai[i] * etai[i](alpha) * ddS(alpha, xi, xii[i]);
        }
        return (0.5 / alpha) * sum + (alpha * alpha) * cosh(alpha);
    }
};

double chnc(double alpha) {
    return cos(alpha) * sinh(alpha) - sin(alpha) * cosh(alpha);
}

auto make_hcc_bf(CrackedBeam cracked_beam) {
    auto chc = [=](auto alpha) mutable {
        auto a11 = cracked_beam.A(alpha, 1);
        auto a12 = cracked_beam.C(alpha, 1);
        auto a21 = cracked_beam.dA(alpha, 1);
        auto a22 = cracked_beam.dC(alpha, 1);
        return a11 * a22 - a12 * a21;
    };
    return chc;
}

auto make_chc_bf(CrackedBeam cracked_beam) {
    auto chc = [=](auto alpha) mutable -> auto {
        auto a11 = cracked_beam.A(alpha, 1) - cracked_beam.C(alpha, 1);
        auto a12 = cracked_beam.B(alpha, 1) - cracked_beam.D(alpha, 1);
        auto a21 = cracked_beam.ddA(alpha, 1) - cracked_beam.ddC(alpha, 1);
        auto a22 = cracked_beam.ddB(alpha, 1) - cracked_beam.ddD(alpha, 1);
        return a11 * a22 - a12 * a21;
    };
    return chc;
}

// TODO: make it not so dummy, lazy ass
auto find_roots(auto fn, Interval interval, size_t discretization) {
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
    return roots;
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

    return [fn, abs_max](auto xi) mutable { return fn(xi) / abs_max; };
}

auto make_eigen_mode_ch(CrackedBeam cracked_beam, int index) {
    auto eigen_frequences_equation = make_chc_bf(cracked_beam);
    auto alphai =
        find_roots(eigen_frequences_equation, Interval{3.7, 11.0}, 10000);
    auto alpha = alphai[index];

    std::cout << "C-H: " << alphai[0] << std::endl;
    auto em = [=](auto xi) mutable {
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

    std::cout << "H-C: " << alphai[1] << std::endl;
    auto em = [=](double xi) mutable {
        auto aux = cracked_beam.A(alpha, xi);
        auto coef = -cracked_beam.A(alpha, 1) / cracked_beam.C(alpha, 1);
        aux += coef * cracked_beam.C(alpha, xi);
        return aux;
    };
    return em;
}

enum BC { CH };

template <enum BC>
struct BeamTraits;

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
struct BeamTraits<BC::CH> {
    BeamTraits(Beam beam) { calculate_roots(); }

    auto Phi(std::size_t m) {
        auto k = roots[m];

        return [&, k](auto xi) {
            return _U(k, xi) - (_S(k, 1.0) / _T(k, 1.0)) * _V(k, xi);
        };
    }

    auto dPhi(std::size_t m) {
        auto k = roots[m];

        return [&, k](auto xi) {
            return k * (_T(k, xi) - (_S(k, 1.0) / _T(k, 1.0)) * _U(k, xi));
        };
    }

private:
    void calculate_roots() {
        roots = find_roots(chnc, Interval{3.7, 11.0}, 1000);
    }

    std::vector<double> roots;
};

template <enum BC>
struct CrackedBeamTraits;

template <>
struct CrackedBeamTraits<BC::CH> {
    CrackedBeamTraits(CrackedBeam cb) : cracked_beam(cb) {
        calculate_roots();
        calculate_lambdai();
        calculate_xi();
        calculate_coefficients();
        calculate_Psi();
        calculate_S_factor();
        calculate_normalization_factors();
    }

    auto S_overline(std::size_t m, std::size_t i) {
        auto w = normalization_factors(m);
        return [&, w, m, i](double xi) { return w * S_overline_raw(m, i, xi); };
    }

    auto dS_overline(std::size_t m, std::size_t i) {
        auto w = normalization_factors(m);
        return
            [&, w, m, i](double xi) { return w * dS_overline_raw(m, i, xi); };
    }

    auto betta(std::size_t m, std::size_t i) {
        auto w = normalization_factors(m);
        return [&, w, m](double xi) { return w * betta_raw(m, xi); };
    }

    auto dbetta(std::size_t m, std::size_t i) {
        auto w = normalization_factors(m);
        return [&, w, m](double xi) { return w * dbetta_raw(m, xi); };
    }

    /// @brief eigen mode function factory
    /// @param m mode number starting from zero
    auto Phi(std::size_t m) {
        auto factor = normalization_factors(m);
        return [&, factor, m](auto xi) { return factor * mode(m, xi); };
    }

private:
    double S_overline_raw(std::size_t m, std::size_t i, double xi) {
        auto alpha = roots[m];
        return S_factor(m, i) * S(alpha, xi, xii(i));
    }

    double dS_overline_raw(std::size_t m, std::size_t i, double xi) {
        auto alpha = roots[m];
        return S_factor(m, i) * dS(alpha, xi, xii(i));
    }

    double betta_raw(std::size_t m, double xi) {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        return sin(alpha * xi) - Psi * cos(alpha * xi) - sinh(alpha * xi) +
               Psi * cosh(alpha * xi);
    }

    double dbetta_raw(std::size_t m, double xi) {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        return alpha * (cos(alpha * xi) + Psi * sin(alpha * xi) -
                        cosh(alpha * xi) + Psi * sinh(alpha * xi));
    }

    auto mode(std::size_t m, auto xi) {
        auto alpha = roots[m];
        auto Psi = Psii[m];
        auto n = cracked_beam.cracks.size();

        auto sum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            auto aux = mui(m, i) - Psi * upsiloni(m, i) - zetai(m, i) +
                       Psi * etai(m, i);
            aux *= lambdai[i];
            sum += S_overline_raw(m, i, xi) * heaviside(xi, xii(i));
        }

        sum = (0.5 / alpha) * sum + betta_raw(m, xi);

        return sum;
    }

    void calculate_roots() {
        auto eq = make_chc_bf(cracked_beam);
        roots = find_roots(eq, Interval{3.7, 11.0}, 1000);
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
                values(row, col) = functions[col](roots[row]);
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
                s_factor(row, col) = aux;
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
                nld::segment{0.0, 1.0});
            factors(i) = 1.0 / distance;
        }
        normalization_factors = factors;
    }

    CrackedBeam cracked_beam;

    std::vector<double> roots;

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

void integrate_eigen_mode(CrackedBeam cracked_beam) {
    auto fem = make_eigen_mode_hc(cracked_beam, 0);
    auto sem = make_eigen_mode_hc(cracked_beam, 1);
    auto mass = [=](auto xi) mutable { return sem(xi) * fem(xi); };
    auto value =
        nld::integrate<nld::gauss_kronrod21>(mass, nld::segment{0.0, 1.0});
    std::cout << "value: " << value << std::endl;
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
    auto value =
        nld::integrate<nld::gauss_kronrod21>(mass, nld::segment{0.0, 1.0});
    std::cout << "value: " << value << std::endl;
}

int main(int argc, char *argv[]) {
    auto material = Material{7800, 2.1e11};
    auto geometry = Geometry{0.177, 0.01, 0.01};
    auto crack = Crack{geometry.length * 0.5, geometry.height * 0.4};
    auto beam = Beam{material, geometry};
    auto cracked_beam = CrackedBeam{beam, std::vector<Crack>{crack}};
    std::cout << "lambdai[0]: " << cracked_beam.lambdai()[0] << std::endl;

    auto fem = make_eigen_mode_ch(cracked_beam, 1);
    auto sem = make_eigen_mode_hc(cracked_beam, 1);
    auto mass = [=](auto xi) mutable { return sem(xi) * fem(xi); };
    CrackedBeamTraits<BC::CH> chcbeam(cracked_beam);
    auto semnc = chcbeam.Phi(1);
    std::cout << "S";

    BeamTraits<BC::CH> traits(beam);

    integrate_eigen_mode(cracked_beam);

    auto N = 200;

    std::vector<double> xii;
    xii.reserve(N + 1);

    std::vector<double> yi;
    yi.reserve(N + 1);

    for (size_t i = 0; i < N + 1; i++) {
        auto xi = (double)i / (double)N;
        auto y = fem(xi);
        xii.push_back(xi);
        yi.push_back(y);
    }

    std::vector<double> yinc;
    yinc.reserve(N + 1);
    for (size_t i = 0; i < N + 1; i++) {
        auto xi = (double)i / (double)N;
        auto y = semnc(xi);
        yinc.push_back(y);
    }

    plt::named_plot("Mode", xii, yi);
    plt::named_plot("Mode_NC", xii, yinc, "r--");
    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::legend();
    plt::show();

    return 0;
}
