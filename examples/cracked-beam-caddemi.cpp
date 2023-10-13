#include <array>
#include <cmath>
#include <iostream>
#include <matplotlibcpp.h>
#include <vector>

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
            roots.push_back(a + dx * i / 2.0);
    }
    return roots;
}

auto make_eigen_mode_ch(CrackedBeam cracked_beam, int index) {
    auto eigen_frequences_equation = make_chc_bf(cracked_beam);
    auto alphai =
        find_roots(eigen_frequences_equation, Interval{3.7, 11.0}, 10000);
    auto alpha = alphai[index];

    auto em = [=](auto xi) mutable {
        auto aux = cracked_beam.A(alpha, xi);
        auto coef = (cracked_beam.A(alpha, 1) - cracked_beam.C(alpha, 1)) /
                    (cracked_beam.B(alpha, 1) - cracked_beam.D(alpha, 1));
        aux -= coef * cracked_beam.B(alpha, xi);
        aux -= cracked_beam.C(alpha, xi);
        aux += coef * cracked_beam.D(alpha, xi);
        return aux;
    };

    return em;
}

auto make_eigen_mode_hc(CrackedBeam cracked_beam, int index) {
    auto eigen_frequences_equation = make_hcc_bf(cracked_beam);
    auto alphai =
        find_roots(eigen_frequences_equation, Interval{3.7, 11.0}, 10000);
    auto alpha = alphai[index];

    auto em = [=](double xi) mutable {
        auto aux = cracked_beam.A(alpha, xi);
        auto coef = -cracked_beam.A(alpha, 1) / cracked_beam.C(alpha, 1);
        aux += coef * cracked_beam.C(alpha, xi);
        return aux;
    };
    return em;
}

int main(int argc, char *argv[]) {
    auto material = Material{7800, 2.1e11};
    auto geometry = Geometry{0.177, 0.01, 0.01};
    auto crack = Crack{geometry.length * 0.5, geometry.height * 0.4};
    auto beam = Beam{material, geometry};
    auto cracked_beam = CrackedBeam{beam, std::vector<Crack>{crack}};

    auto first_eigen_mode = make_eigen_mode_ch(cracked_beam, 2);

    auto N = 200;

    std::vector<double> xii;
    xii.reserve(N + 1);

    std::vector<double> yi;
    yi.reserve(N + 1);

    for (size_t i = 0; i < N + 1; i++) {
        auto xi = (double)i / (double)N;
        auto y = first_eigen_mode(xi);
        xii.push_back(xi);
        yi.push_back(y);

        std::cout << "xi: " << xi << std::endl;
        std::cout << "y: " << y << std::endl;
    }

    plt::named_plot("Mode", xii, yi);
    plt::ylabel(R"($A_1$)");
    plt::xlabel(R"($\Omega$)");
    plt::legend();
    plt::show();

    return 0;
}
