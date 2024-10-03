#pragma once
#include <cassert>
#include <complex>
#include <functional>
#include <nld/core.hpp>
#include <ranges>
#include <vector>

namespace nld {

namespace detail {
inline auto build_new_matricies(const Eigen::MatrixXcd &M_0,
                                const Eigen::MatrixXcd &M_1,
                                const Eigen::MatrixXcd &M_2, double alpha)
    -> std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> {
    Eigen::MatrixXcd M_0_new = M_0 + alpha * M_1 + alpha * alpha * M_2;
    Eigen::MatrixXcd M_1_new = M_1 + 2 * alpha * M_2;
    Eigen::MatrixXcd M_2_new = M_2;

    return {M_0_new, M_1_new, M_2_new};
}

inline auto liniarized(const Eigen::MatrixXcd &M_0, const Eigen::MatrixXcd &M_1,
                       const Eigen::MatrixXcd &M_2,
                       double omega) -> Eigen::MatrixXcd {
    auto size = M_0.rows();
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(2 * size, 2 * size);
    Eigen::MatrixXcd M_0_inv = M_0.inverse();
    A.block(0, 0, size, size) = -M_0_inv * M_1;
    A.block(0, size, size, size) = -M_0_inv * M_2;
    A.block(size, 0, size, size) = Eigen::MatrixXcd::Identity(size, size);
    A = A - omega * Eigen::MatrixXcd::Identity(2 * size, 2 * size);
    return A;
}

inline auto bialternate_element(const Eigen::MatrixXcd &A, int p, int q, int r,
                                int s) -> std::complex<double> {
    using namespace std::complex_literals;
    if (r == q)
        return -A(p, s);
    if (r != p && s == q)
        return A(p, r);
    if (r == p && s == q)
        return A(p, p) + A(q, q);
    if (r == p && s != q)
        return A(q, s);
    if (s == p)
        return -A(q, r);
    else
        return 0.0 + 0.0i;
}

inline auto bialternate_sum(const Eigen::MatrixXcd &A) -> Eigen::MatrixXcd {
    auto n = A.rows();
    auto bn = int(1.0 / 2.0 * (n - 1) * n);
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(bn, bn);

    for (int p = 1; p < n; ++p) {
        for (int q = 0; q < p; ++q) {
            for (int r = 1; r < n; ++r) {
                for (int s = 0; s < r; ++s) {
                    auto row = int(1.0 / 2.0 * p * (p - 1) + q);
                    auto col = int(1.0 / 2.0 * r * (r - 1) + s);
                    B(row, col) = bialternate_element(A, p, q, r, s);
                }
            }
        }
    }

    return B;
}

inline auto lyapunov_element(const Eigen::MatrixXcd &A, int p, int q, int r,
                             int s) -> std::complex<double> {
    if (p > q) {
        if (r == q && s < q)
            return A(p, s);
        if (r != p && s == q)
            return A(p, r);
        if (r == p && s == q)
            return A(p, p) + A(q, q);
        if (r == p && s != q)
            return A(q, s);
        if (r > p && s == p)
            return A(q, r);
        else
            return 0;
    }

    if (p == q) {
        if (r == p && s < p)
            return 2.0 * A(p, s);
        if (r == p && s == p)
            return 2.0 * A(p, p);
        if (r > p && s == p)
            return 2.0 * A(p, r);
        else
            return 0;
    }

    return 0;
}

inline auto lyapunov_matrix(const Eigen::MatrixXcd &A) -> Eigen::MatrixXcd {
    auto n = A.rows();
    auto bn = int(1.0 / 2.0 * n * (n + 1));
    Eigen::MatrixXcd B = Eigen::MatrixXcd::Zero(bn, bn);

    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    auto row = int(1.0 / 2.0 * p * (p + 1) + q);
                    auto col = int(1.0 / 2.0 * r * (r + 1) + s);
                    B(row, col) = lyapunov_element(A, p, q, r, s);
                }
            }
        }
    }

    return B;
}

} // namespace detail

enum class resonance_type {
    harmonic,
    subharmonic,
    combinational,
    all,
};

using matrix_maker_fn = std::function<nld::matrix_xd(double)>;
using matrix_maker_type = std::vector<matrix_maker_fn>;

struct generalized_bolotin_method {
    generalized_bolotin_method(const matrix_maker_type &Cnf,
                               const matrix_maker_type &Knf, int m,
                               resonance_type rt, double alpha = 0.05)
        : Cnf(Cnf), Knf(Knf), m(m), rt(rt), alpha(alpha) {}

    template <typename Vector>
    auto operator()(const Vector &variables) const -> Vector {
        assert(variables.size() == 2 && "wrong number of variables");

        auto lambda = variables(0);
        auto omega = variables(1);

        Eigen::MatrixXcd M_0, M_1, M_2;

        switch (rt) {
        case resonance_type::harmonic:
            std::tie(M_0, M_1, M_2) =
                harmonic_resonance_matricies(lambda, omega);
            break;
        case resonance_type::subharmonic:
            std::tie(M_0, M_1, M_2) =
                subharmonic_resonance_matricies(lambda, omega);
            break;
        case resonance_type::combinational:
            std::tie(M_0, M_1, M_2) =
                combinational_resonance_matricies(lambda, omega);
            break;
        case resonance_type::all:
            std::tie(M_0, M_1, M_2) = all_resonance_matricies(lambda, omega);
            break;
        default:
            throw std::runtime_error("Not implemented");
        }

        if (rt != resonance_type::combinational && rt != resonance_type::all) {
            auto [M_0_hat, M_1_hat, M_2_hat] =
                detail::build_new_matricies(M_0, M_1, M_2, alpha);

            auto A = detail::liniarized(M_0_hat, M_1_hat, M_2_hat, omega);

            auto det = A.determinant();

            Vector result(1);
            result(0) = det.real();

            return result;
        } else if (rt == resonance_type::combinational) {
            // Eigen::MatrixXcd A =
            //     M_0 + 1.0 / omega * M_1 + 1.0 / (omega * omega) * M_2;
            // auto det = A.determinant();
            //
            // Vector result(1);
            // auto real = det.real();
            // // std::cout << "real: " << real << std::endl;
            // auto log_real = std::round(std::log(std::abs(real)));
            // A = A / omega;
            // det = A.determinant();
            // real = det.real();
            // // real = real / std::exp(log_real + 10);
            // std::cout << "real scaled: " << real << std::endl;
            // result(0) = det.real();
            //
            // return result;
            auto [M_0_hat, M_1_hat, M_2_hat] =
                detail::build_new_matricies(M_0, M_1, M_2, alpha);

            auto A = detail::liniarized(M_0_hat, M_1_hat, M_2_hat, omega);
            A = A / sqrt(1.2 * omega * m);

            auto det = A.determinant();

            Vector result(1);
            result(0) = det.real();

            std::cout << "det: " << det.real() << std::endl;

            return result;
        } else {
            auto [M_0_hat, M_1_hat, M_2_hat] =
                detail::build_new_matricies(M_0, M_1, M_2, alpha);

            auto A = detail::liniarized(M_0_hat, M_1_hat, M_2_hat, omega);
            std::cout << "A size: " << A.rows() << " " << A.cols() << std::endl;
            A = A / sqrt(omega * 0.9 * log(double(A.rows())));
            auto det = A.determinant();

            Vector result(1);
            result(0) = det.real();

            std::cout << "det: " << det.real() << std::endl;

            return result;
        }
    }

    template <typename At, typename V>
    auto jacobian(At &u, V &v) const -> nld::matrix_xd {
        // Create a square matrix of size u.size()
        nld::matrix_xd J(u.size(), u.size());
        auto fn = [this](const auto &u) -> double {
            return this->operator()(u)(0);
        };
        auto grad = nld::math::detail::gradient(fn, u, v(0));
        J.row(0) = grad;

        return J;
    }

    [[nodiscard]] auto underlying_function() const noexcept { return *this; }

    [[nodiscard]] auto
    from_omega_hat_to_omega(double omega_hat) const noexcept {
        return omega_hat / (1 + omega_hat * alpha);
    }

    [[nodiscard]] auto from_omega_to_omega_hat(double omega) const noexcept {
        return omega / (1 - omega * alpha);
    }

private:
    auto harmonic_resonance_matricies(double lambda, double omega) const
        -> std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> {
        using namespace std::complex_literals;

        int h_all = Cnf.size();
        int h = std::size_t((h_all - 1) / 2);

        std::vector<nld::matrix_xd> Cn;
        Cn.reserve(h_all);
        std::vector<nld::matrix_xd> Kn;
        Kn.reserve(h_all);

        std::ranges::transform(Cnf, std::back_inserter(Cn),
                               [&](const auto &fn) { return fn(lambda); });
        std::ranges::transform(Knf, std::back_inserter(Kn),
                               [&](const auto &fn) { return fn(lambda); });

        auto n = Cn[0].rows();
        auto size = 2 * n * m + n;

        Eigen::MatrixXcd F_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_2 = Eigen::MatrixXcd::Zero(size, size);

        for (int k = -m; k <= m; ++k) {
            int k_aux = k + m;
            Eigen::MatrixXd F_0_ = -k * k * Eigen::MatrixXd::Identity(n, n);

            for (int p = -h; p <= h; ++p) {
                int q = k - p;
                double delta_kq = k == q ? 1 : 0;
                int q_aux = q + m;

                if (q < -m || q > m)
                    continue;

                int row = k_aux * n;
                int row_begin = row;

                int col = q_aux * n;
                int col_begin = col;

                F_0.block(row_begin, col_begin, n, n) += F_0_ * delta_kq;
                F_1.block(row_begin, col_begin, n, n) +=
                    1.0i * double(q) * Cn[h + p];
                F_2.block(row_begin, col_begin, n, n) += Kn[h + p];
            }
        }

        return {F_0, F_1, F_2};
    }

    auto subharmonic_resonance_matricies(double lambda, double omega) const
        -> std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> {
        using namespace std::complex_literals;

        int h_all = Cnf.size();
        int h = std::size_t((h_all - 1) / 2);

        std::vector<nld::matrix_xd> Cn;
        Cn.reserve(h_all);
        std::vector<nld::matrix_xd> Kn;
        Kn.reserve(h_all);

        std::ranges::transform(Cnf, std::back_inserter(Cn),
                               [&](const auto &fn) { return fn(lambda); });
        std::ranges::transform(Knf, std::back_inserter(Kn),
                               [&](const auto &fn) { return fn(lambda); });

        auto n = Cn[0].rows();
        auto size = 2 * n * m + n;

        Eigen::MatrixXcd M_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd M_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd M_2 = Eigen::MatrixXcd::Zero(size, size);

        for (int k = -m; k <= m; ++k) {
            int k_aux = k + m;
            Eigen::MatrixXcd E_0_ =
                2.0 * 1.0i * double(k) * Eigen::MatrixXd::Identity(n, n);
            Eigen::MatrixXd F_0_ = -k * k * Eigen::MatrixXd::Identity(n, n);

            for (int p = -h; p <= h; ++p) {
                int q = k - p;
                double delta_kq = k == q ? 1 : 0;
                int q_aux = q + m;

                if (q < -m || q > m)
                    continue;

                int row = k_aux * n;
                int row_begin = row;

                int col = q_aux * n;
                int col_begin = col;

                Eigen::MatrixXcd F_1_ = 1.0i * double(q) * Cn[h + p];
                Eigen::MatrixXcd F_2_ = Kn[h + p];
                Eigen::MatrixXcd E_1_ = Cn[h + p];

                M_0.block(row_begin, col_begin, n, n) +=
                    F_0_ * delta_kq + 1.0 / 2 * 1.0i * E_0_ * delta_kq -
                    1.0 / 4.0 * Eigen::MatrixXd::Identity(n, n) * delta_kq;
                M_1.block(row_begin, col_begin, n, n) +=
                    F_1_ + 1.0 / 2 * 1.0i * E_1_;
                M_2.block(row_begin, col_begin, n, n) += F_2_;
            }
        }

        return {M_0, M_1, M_2};
    }

    auto combinational_resonance_matricies(double lambda, double omega) const
        -> std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> {
        using namespace std::complex_literals;

        int h_all = Cnf.size();
        int h = std::size_t((h_all - 1) / 2);

        std::vector<nld::matrix_xd> Cn;
        Cn.reserve(h_all);
        std::vector<nld::matrix_xd> Kn;
        Kn.reserve(h_all);

        std::ranges::transform(Cnf, std::back_inserter(Cn),
                               [&](const auto &fn) { return fn(lambda); });
        std::ranges::transform(Knf, std::back_inserter(Kn),
                               [&](const auto &fn) { return fn(lambda); });

        auto n = Cn[0].rows();
        auto size = 2 * n * m + n;

        Eigen::MatrixXcd E_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd E_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_2 = Eigen::MatrixXcd::Zero(size, size);

        for (int k = -m; k <= m; ++k) {
            int k_aux = k + m;
            Eigen::MatrixXcd E_0_ =
                2.0 * 1.0i * double(k) * Eigen::MatrixXd::Identity(n, n);
            Eigen::MatrixXd F_0_ = -k * k * Eigen::MatrixXd::Identity(n, n);

            for (int p = -h; p <= h; ++p) {
                int q = k - p;
                double delta_kq = k == q ? 1 : 0;
                int q_aux = q + m;

                if (q < -m || q > m)
                    continue;

                int row = k_aux * n;
                int row_begin = row;

                int col = q_aux * n;
                int col_begin = col;

                Eigen::MatrixXcd F_1_ = 1.0i * double(q) * Cn[h + p];
                Eigen::MatrixXcd F_2_ = Kn[h + p];
                Eigen::MatrixXcd E_1_ = Cn[h + p];

                E_0.block(row_begin, col_begin, n, n) = E_0_ * delta_kq;
                F_0.block(row_begin, col_begin, n, n) = F_0_ * delta_kq;
                E_1.block(row_begin, col_begin, n, n) = E_1_;
                F_1.block(row_begin, col_begin, n, n) = F_1_;
                F_2.block(row_begin, col_begin, n, n) = F_2_;
            }
        }

        Eigen::MatrixXcd U_0 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);
        Eigen::MatrixXcd U_1 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);
        Eigen::MatrixXcd U_2 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);

        U_0.block(0, 0, size, size) = -E_0;
        U_0.block(0, size, size, size) = -F_0;
        U_0.block(size, 0, size, size) = Eigen::MatrixXcd::Identity(size, size);

        U_1.block(0, 0, size, size) = -E_1;
        U_1.block(0, size, size, size) = -F_1;

        U_2.block(0, size, size, size) = -F_2;

        Eigen::MatrixXcd B_0 = detail::bialternate_sum(U_0);
        Eigen::MatrixXcd B_1 = detail::bialternate_sum(U_1);
        Eigen::MatrixXcd B_2 = detail::bialternate_sum(U_2);

        return {B_0, B_1, B_2};
    }

    auto all_resonance_matricies(double lambda, double omega) const
        -> std::tuple<Eigen::MatrixXcd, Eigen::MatrixXcd, Eigen::MatrixXcd> {
        using namespace std::complex_literals;

        int h_all = Cnf.size();
        int h = std::size_t((h_all - 1) / 2);

        std::vector<nld::matrix_xd> Cn;
        Cn.reserve(h_all);
        std::vector<nld::matrix_xd> Kn;
        Kn.reserve(h_all);

        std::ranges::transform(Cnf, std::back_inserter(Cn),
                               [&](const auto &fn) { return fn(lambda); });
        std::ranges::transform(Knf, std::back_inserter(Kn),
                               [&](const auto &fn) { return fn(lambda); });

        auto n = Cn[0].rows();
        auto size = 2 * n * m + n;

        Eigen::MatrixXcd E_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_0 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd E_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_1 = Eigen::MatrixXcd::Zero(size, size);
        Eigen::MatrixXcd F_2 = Eigen::MatrixXcd::Zero(size, size);

        for (int k = -m; k <= m; ++k) {
            int k_aux = k + m;
            Eigen::MatrixXcd E_0_ =
                2.0 * 1.0i * double(k) * Eigen::MatrixXd::Identity(n, n);
            Eigen::MatrixXd F_0_ = -k * k * Eigen::MatrixXd::Identity(n, n);

            for (int p = -h; p <= h; ++p) {
                int q = k - p;
                double delta_kq = k == q ? 1 : 0;
                int q_aux = q + m;

                if (q < -m || q > m)
                    continue;

                int row = k_aux * n;
                int row_begin = row;

                int col = q_aux * n;
                int col_begin = col;

                Eigen::MatrixXcd F_1_ = 1.0i * double(q) * Cn[h + p];
                Eigen::MatrixXcd F_2_ = Kn[h + p];
                Eigen::MatrixXcd E_1_ = Cn[h + p];

                E_0.block(row_begin, col_begin, n, n) = E_0_ * delta_kq;
                F_0.block(row_begin, col_begin, n, n) = F_0_ * delta_kq;
                E_1.block(row_begin, col_begin, n, n) = E_1_;
                F_1.block(row_begin, col_begin, n, n) = F_1_;
                F_2.block(row_begin, col_begin, n, n) = F_2_;
            }
        }

        Eigen::MatrixXcd U_0 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);
        Eigen::MatrixXcd U_1 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);
        Eigen::MatrixXcd U_2 = Eigen::MatrixXcd::Zero(2 * size, 2 * size);

        U_0.block(0, 0, size, size) = -E_0;
        U_0.block(0, size, size, size) = -F_0;
        U_0.block(size, 0, size, size) = Eigen::MatrixXcd::Identity(size, size);

        U_1.block(0, 0, size, size) = -E_1;
        U_1.block(0, size, size, size) = -F_1;

        U_2.block(0, size, size, size) = -F_2;

        Eigen::MatrixXcd B_0 = detail::lyapunov_matrix(U_0);
        Eigen::MatrixXcd B_1 = detail::lyapunov_matrix(U_1);
        Eigen::MatrixXcd B_2 = detail::lyapunov_matrix(U_2);

        return {B_0, B_1, B_2};
    }

    matrix_maker_type Cnf;
    matrix_maker_type Knf;
    int m;
    resonance_type rt;
    double alpha;
};

} // namespace nld
