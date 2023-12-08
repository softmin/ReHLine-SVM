#ifndef REHLINE_H
#define REHLINE_H

#include <vector>
#include <numeric>
#include <random>
#include <type_traits>
#include <iostream>
#include <Eigen/Core>

namespace rehline {

// ========================= Internal utility functions ========================= //
namespace internal {

// A simple wrapper of existing RNG
template <typename Index = int>
class SimpleRNG
{
private:
    std::mt19937 m_rng;

public:
    // Set seed
    void seed(Index seed) { m_rng.seed(seed); }

    // Used in random_shuffle(), generating a random integer from {0, 1, ..., i-1}
    Index operator()(Index i)
    {
        return Index(m_rng() % i);
    }
};

// Randomly shuffle a vector
//
// On Mac, std::random_shuffle() uses a "backward" implementation,
// which leads to different results from Windows and Linux
// Therefore, we use a consistent implementation based on GCC code
template <typename RandomAccessIterator, typename RandomNumberGenerator>
void random_shuffle(RandomAccessIterator first, RandomAccessIterator last, RandomNumberGenerator& gen)
{
    if(first == last)
        return;
    for(RandomAccessIterator i = first + 1; i != last; ++i)
    {
        RandomAccessIterator j = first + gen((i - first) + 1);
        if(i != j)
            std::iter_swap(i, j);
    }
}

// Reset the free variable set to [0, 1, ..., n-1] (if the variables form a vector)
template <typename Index = int>
void reset_fv_set(std::vector<Index>& fvset, std::size_t n)
{
    fvset.resize(n);
    // Fill the vector with 0, 1, ..., n-1
    std::iota(fvset.begin(), fvset.end(), Index(0));
}

// Reset the free variable set to [(0, 0), (0, 1), ..., (n-1, m-2), (n-1, m-1)] (if the variables form a matrix)
template <typename Index = int>
void reset_fv_set(std::vector<std::pair<Index, Index>>& fvset, std::size_t n, std::size_t m)
{
    fvset.resize(n * m);
    for(std::size_t i = 0; i < n * m; i++)
        fvset[i] = std::make_pair(i % n, i / n);
}


}  // namespace internal
// ========================= Internal utility functions ========================= //



// Dimensions of the matrices involved
// - Input
//   * X        : [n x d]
//   * U, V     : [L x n]
//   * S, T, Tau: [H x n]
//   * A        : [K x d]
//   * b        : [K]
// - Pre-computed
//   * r: [n]
//   * p: [K]
// - Primal
//   * beta: [d]
// - Dual
//   * xi    : [K]
//   * Lambda: [L x n]
//   * Gamma : [H x n]

// Results of the optimization algorithm
template <typename Matrix = Eigen::MatrixXd, typename Index = int>
struct ReHLineResult
{
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Vector              beta;           // Primal variable
    Vector              xi;             // Dual variables
    Matrix              Lambda;         // Dual variables
    Matrix              Gamma;          // Dual variables
    Index               niter;          // Number of iterations
    std::vector<Scalar> dual_objfns;    // Recorded dual objective function values
    std::vector<Scalar> primal_objfns;  // Recorded primal objective function values
};

// The main ReHLine solver
// "Matrix" is the type of input data matrix, can be row-majored or column-majored
template <typename Matrix = Eigen::MatrixXd, typename Index = int>
class ReHLineSolver
{
private:
    using Scalar = typename Matrix::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstRefMat = Eigen::Ref<const Matrix>;
    using ConstRefVec = Eigen::Ref<const Vector>;

    // We really want some matrices to be row-majored, since they can be more
    // efficient in certain matrix operations, for example X.row(i).dot(v)
    //
    // If the data Matrix is already row-majored, we save a const reference;
    // otherwise we make a copy
    using RMatrix = typename std::conditional<
        Matrix::IsRowMajor,
        Eigen::Ref<const Matrix>,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >::type;

    // RNG
    internal::SimpleRNG<Index> m_rng;

    // Dimensions
    const Index m_n;
    const Index m_d;
    const Index m_L;
    const Index m_H;
    const Index m_K;

    // Input matrices and vectors
    RMatrix     m_X;
    ConstRefMat m_U;
    ConstRefMat m_V;
    ConstRefMat m_S;
    ConstRefMat m_T;
    ConstRefMat m_Tau;
    RMatrix     m_A;
    ConstRefVec m_b;

    // Pre-computed
    Vector m_gk_denom;   // ||a[k]||^2
    Matrix m_gli_denom;  // (u[li] * ||x[i]||)^2
    Matrix m_ghi_denom;  // (s[hi] * ||x[i]||)^2 + 1

    // Primal variable
    Vector m_beta;

    // Dual variables
    Vector m_xi;
    Matrix m_Lambda;
    Matrix m_Gamma;

    // Free variable sets
    std::vector<Index> m_fv_feas;
    std::vector<std::pair<Index, Index>> m_fv_relu;
    std::vector<std::pair<Index, Index>> m_fv_rehu;

    // =================== Initialization functions =================== //

    // Compute the primal variable beta from dual variables
    // beta = A'xi - U3 * vec(Lambda) - S3 * vec(Gamma)
    // A can be empty, one of U and V may be empty
    inline void set_primal()
    {
        // Initialize beta to zero
        m_beta.setZero();

        // First term
        if (m_K > 0)
            m_beta.noalias() = m_A.transpose() * m_xi;

        // [n x 1]
        Vector LHterm = Vector::Zero(m_n);
        if (m_L > 0)
            LHterm.noalias() = m_U.cwiseProduct(m_Lambda).colwise().sum().transpose();
        // [n x 1]
        if (m_H > 0)
            LHterm.noalias() += m_S.cwiseProduct(m_Gamma).colwise().sum().transpose();

        m_beta.noalias() -= m_X.transpose() * LHterm;
    }

    // =================== Evaluating objection function =================== //

    // Compute the primal objective function value
    inline Scalar primal_objfn() const
    {
        Scalar result = Scalar(0);
        const Vector Xbeta = m_X * m_beta;
        // ReLU part
        if (m_L > 0)
        {
            result += (m_U.cwiseProduct(Xbeta.transpose().replicate(1, m_L)) +
                m_V).cwiseMax(Scalar(0)).sum();
        }
        // ReHU part
        if (m_H > 0)
        {
            const Matrix z = (m_S.cwiseProduct(Xbeta.transpose().replicate(1, m_H)) +
                m_T).cwiseMax(Scalar(0));
            result += (z.array() <= m_Tau.array()).select(
                z.array().square() * Scalar(0.5),
                m_Tau.array() * (z.array() - m_Tau.array() * Scalar(0.5))
            ).sum();
        }
        // Quadratic term
        result += Scalar(0.5) * m_beta.squaredNorm();
        return result;
    }

    // Compute the dual objective function value
    inline Scalar dual_objfn() const
    {
        // A' * xi, [d x 1], A[K x d] may be empty
        Vector Atxi = Vector::Zero(m_d);
        if (m_K > 0)
            Atxi.noalias() = m_A.transpose() * m_xi;
        // U3 * vec(Lambda), [n x 1], U[L x n] may be empty
        Vector UL(m_n), U3L = Vector::Zero(m_d);
        if (m_L > 0)
        {
            UL.noalias() = m_U.cwiseProduct(m_Lambda).colwise().sum().transpose();
            U3L.noalias() = m_X.transpose() * UL;
        }
        // S3 * vec(Gamma), [n x 1], S[H x n] may be empty
        Vector SG(m_n), S3G = Vector::Zero(m_d);
        if (m_H > 0)
        {
            SG.noalias() = m_S.cwiseProduct(m_Gamma).colwise().sum().transpose();
            S3G.noalias() = m_X.transpose() * SG;
        }

        // Compute dual objective function value
        Scalar obj = Scalar(0);
        // If K = 0, all terms that depend on A, xi, or b will be zero
        if (m_K > 0)
        {
            // 0.5 * ||Atxi||^2 - Atxi' * U3L - Atxi' * S3G + xi' * b
            const Scalar Atxi_U3L = (m_L > 0) ? (Atxi.dot(U3L)) : Scalar(0);
            const Scalar Atxi_S3G = (m_H > 0) ? (Atxi.dot(S3G)) : Scalar(0);
            obj += Scalar(0.5) * Atxi.squaredNorm() - Atxi_U3L - Atxi_S3G + m_xi.dot(m_b);
        }
        // If L = 0, all terms that depend on U, V, or Lambda will be zero
        if (m_L > 0)
        {
            // 0.5 * ||U3L||^2 + U3L' * S3G - tr(Lambda * V')
            const Scalar U3L_S3G = (m_H > 0) ? (U3L.dot(S3G)) : Scalar(0);
            obj += Scalar(0.5) * U3L.squaredNorm() + U3L_S3G -
                m_Lambda.cwiseProduct(m_V).sum();
        }
        // If H = 0, all terms that depend on S, T, or Gamma will be zero
        if (m_H > 0)
        {
            // 0.5 * ||S3G||^2 + 0.5 * ||Gamma||^2 - tr(Gamma * T')
            obj += Scalar(0.5) * S3G.squaredNorm() + Scalar(0.5) * m_Gamma.squaredNorm() -
                m_Gamma.cwiseProduct(m_T).sum();
        }

        return obj;
    }

    // =================== Updating functions (sequential) =================== //

    // Update xi and beta
    inline void update_xi_beta()
    {
        if (m_K < 1)
            return;

        for (Index k = 0; k < m_K; k++)
        {
            const Scalar xi_k = m_xi[k];

            // Compute g_k
            const Scalar g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // Compute new xi_k
            const Scalar candid = xi_k - g_k / m_gk_denom[k];
            const Scalar newxi = std::max(Scalar(0), candid);
            // Update xi and beta
            m_xi[k] = newxi;
            m_beta.noalias() += (newxi - xi_k) * m_A.row(k).transpose();
        }
    }

    // Update Lambda and beta
    inline void update_Lambda_beta()
    {
        if (m_L < 1)
            return;

        for (Index i = 0; i < m_n; i++)
        {
            for (Index l = 0; l < m_L; l++)
            {
                const Scalar u_li = m_U(l, i);
                const Scalar v_li = m_V(l, i);
                const Scalar lambda_li = m_Lambda(l, i);

                // Compute g_li
                const Scalar g_li = -(u_li * m_X.row(i).dot(m_beta) + v_li);
                // Compute new lambda_li
                const Scalar candid = lambda_li - g_li / m_gli_denom(l, i);
                const Scalar newl = std::max(Scalar(0), std::min(Scalar(1), candid));
                // Update Lambda and beta
                m_Lambda(l, i) = newl;
                m_beta.noalias() -= (newl - lambda_li) * u_li * m_X.row(i).transpose();
            }
        }
    }

    // Update Gamma, and beta
    inline void update_Gamma_beta()
    {
        if (m_H < 1)
            return;

        for (Index i = 0; i < m_n; i++)
        {
            for (Index h = 0; h < m_H; h++)
            {
                // tau_hi can be Inf
                const Scalar tau_hi = m_Tau(h, i);
                const Scalar gamma_hi = m_Gamma(h, i);
                const Scalar s_hi = m_S(h, i);
                const Scalar t_hi = m_T(h, i);

                // Compute g_hi
                const Scalar g_hi = gamma_hi - (s_hi * m_X.row(i).dot(m_beta) + t_hi);
                // Compute new gamma_hi
                const Scalar candid = gamma_hi - g_hi / m_ghi_denom(h, i);
                const Scalar newg = std::max(Scalar(0), std::min(tau_hi, candid));
                // Update Gamma and beta
                m_Gamma(h, i) = newg;
                m_beta.noalias() -= (newg - gamma_hi) * s_hi * m_X.row(i).transpose();
            }
        }
    }

    // =================== Updating functions (free variable set) ================ //

    // Determine whether to shrink xi, and compute the projected gradient (PG)
    // Shrink if xi=0 and grad>ub
    // PG is zero if xi=0 and grad>=0
    inline bool pg_xi(Scalar xi, Scalar grad, Scalar ub, Scalar& pg) const
    {
        pg = (xi == Scalar(0) && grad >= Scalar(0)) ? Scalar(0) : grad;
        const bool shrink = (xi == Scalar(0)) && (grad > ub);
        return shrink;
    }
    // Update xi and beta
    // Overloaded version based on free variable set
    inline void update_xi_beta(std::vector<Index>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_K < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<Index> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking threshold
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto k: fv_set)
        {
            const Scalar xi_k = m_xi[k];

            // Compute g_k
            const Scalar g_k = m_A.row(k).dot(m_beta) + m_b[k];
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_xi(xi_k, g_k, ub, pg);
            if (shrink)
               continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new xi_k
            const Scalar candid = xi_k - g_k / m_gk_denom[k];
            const Scalar newxi = std::max(Scalar(0), candid);
            // Update xi and beta
            m_xi[k] = newxi;
            m_beta.noalias() += (newxi - xi_k) * m_A.row(k).transpose();

            // Add to new free variable set
            new_set.push_back(k);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink lambda, and compute the projected gradient (PG)
    // Shrink if (lambda=0 and grad>ub) or (lambda=1 and grad<lb)
    // PG is zero if (lambda=0 and grad>=0) or (lambda=1 and grad<=0)
    inline bool pg_lambda(Scalar lambda, Scalar grad, Scalar lb, Scalar ub, Scalar& pg) const
    {
        pg = ((lambda == Scalar(0) && grad >= Scalar(0)) || (lambda == Scalar(1) && grad <= Scalar(0))) ?
             Scalar(0) :
             grad;
        const bool shrink = (lambda == Scalar(0) && grad > ub) || (lambda == Scalar(1) && grad < lb);
        return shrink;
    }
    // Update Lambda and beta
    // Overloaded version based on free variable set
    inline void update_Lambda_beta(std::vector<std::pair<Index, Index>>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_L < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<std::pair<Index, Index>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar lb = (min_pg < Scalar(0)) ? min_pg : -Inf;
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto rc: fv_set)
        {
            const Index l = rc.first;
            const Index i = rc.second;

            const Scalar u_li = m_U(l, i);
            const Scalar v_li = m_V(l, i);
            const Scalar lambda_li = m_Lambda(l, i);

            // Compute g_li
            const Scalar g_li = -(u_li * m_X.row(i).dot(m_beta) + v_li);
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_lambda(lambda_li, g_li, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new lambda_li
            const Scalar candid = lambda_li - g_li / m_gli_denom(l, i);
            const Scalar newl = std::max(Scalar(0), std::min(Scalar(1), candid));
            // Update Lambda and beta
            m_Lambda(l, i) = newl;
            m_beta.noalias() -= (newl - lambda_li) * u_li * m_X.row(i).transpose();

            // Add to new free variable set
            new_set.emplace_back(l, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

    // Determine whether to shrink gamma, and compute the projected gradient (PG)
    // Shrink if (gamma=0 and grad>ub) or (lambda=tau and grad<lb)
    // PG is zero if (lambda=0 and grad>=0) or (lambda=1 and grad<=0)
    inline bool pg_gamma(Scalar gamma, Scalar grad, Scalar tau, Scalar lb, Scalar ub, Scalar& pg) const
    {
        pg = ((gamma == Scalar(0) && grad >= Scalar(0)) || (gamma == tau && grad <= Scalar(0))) ?
             Scalar(0) :
             grad;
        const bool shrink = (gamma == Scalar(0) && grad > ub) || (gamma == tau && grad < lb);
        return shrink;
    }
    // Update Gamma and beta
    // Overloaded version based on free variable set
    inline void update_Gamma_beta(std::vector<std::pair<Index, Index>>& fv_set, Scalar& min_pg, Scalar& max_pg)
    {
        if (m_H < 1)
            return;

        // Permutation
        internal::random_shuffle(fv_set.begin(), fv_set.end(), m_rng);
        // New free variable set
        std::vector<std::pair<Index, Index>> new_set;
        new_set.reserve(fv_set.size());

        // Compute shrinking thresholds
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        const Scalar lb = (min_pg < Scalar(0)) ? min_pg : -Inf;
        const Scalar ub = (max_pg > Scalar(0)) ? max_pg : Inf;
        // Compute minimum and maximum projected gradient (PG) for this round
        min_pg = Inf;
        max_pg = -Inf;
        for (auto rc: fv_set)
        {
            const Index h = rc.first;
            const Index i = rc.second;

            // tau_hi can be Inf
            const Scalar tau_hi = m_Tau(h, i);
            const Scalar gamma_hi = m_Gamma(h, i);
            const Scalar s_hi = m_S(h, i);
            const Scalar t_hi = m_T(h, i);

            // Compute g_hi
            const Scalar g_hi = gamma_hi - (s_hi * m_X.row(i).dot(m_beta) + t_hi);
            // PG and shrink
            Scalar pg;
            const bool shrink = pg_gamma(gamma_hi, g_hi, tau_hi, lb, ub, pg);
            if (shrink)
                continue;

            // Update PG bounds
            max_pg = std::max(max_pg, pg);
            min_pg = std::min(min_pg, pg);
            // Compute new gamma_hi
            const Scalar candid = gamma_hi - g_hi / m_ghi_denom(h, i);
            const Scalar newg = std::max(Scalar(0), std::min(tau_hi, candid));
            // Update Gamma and beta
            m_Gamma(h, i) = newg;
            m_beta.noalias() -= (newg - gamma_hi) * s_hi * m_X.row(i).transpose();

            // Add to new free variable set
            new_set.emplace_back(h, i);
        }

        // Update free variable set
        fv_set.swap(new_set);
    }

public:
    ReHLineSolver(ConstRefMat X, ConstRefMat U, ConstRefMat V,
                  ConstRefMat S, ConstRefMat T, ConstRefMat Tau,
                  ConstRefMat A, ConstRefVec b) :
        m_n(X.rows()), m_d(X.cols()), m_L(U.rows()), m_H(S.rows()), m_K(A.rows()),
        m_X(X), m_U(U), m_V(V), m_S(S), m_T(T), m_Tau(Tau), m_A(A), m_b(b),
        m_gk_denom(m_K), m_gli_denom(m_L, m_n), m_ghi_denom(m_H, m_n),
        m_beta(m_d),
        m_xi(m_K), m_Lambda(m_L, m_n), m_Gamma(m_H, m_n)
    {
        // A [K x d], K can be zero
        if (m_K > 0)
            m_gk_denom.noalias() = m_A.rowwise().squaredNorm();

        Vector xi2 = m_X.rowwise().squaredNorm();
        if (m_L > 0)
        {
            m_gli_denom.array() = m_U.array().square().rowwise() * xi2.transpose().array();
        }

        if (m_H > 0)
        {
            m_ghi_denom.array() = m_S.array().square().rowwise() * xi2.transpose().array() + Scalar(1);
        }
    }

    // Initialize primal and dual variables
    inline void init_params()
    {
        // xi >= 0, initialized to be 1
        if (m_K > 0)
            m_xi.fill(Scalar(1));

        // Each element of Lambda satisfies 0 <= lambda_li <= 1,
        // and we use 0.5 to initialize Lambda
        if (m_L > 0)
            m_Lambda.fill(Scalar(0.5));

        // Each element of Gamma satisfies 0 <= gamma_hi <= tau_hi,
        // and we use min(0.5 * tau_hi, 1) to initialize (tau_hi can be Inf)
        if (m_H > 0)
        {
            m_Gamma.noalias() = (Scalar(0.5) * m_Tau).cwiseMin(Scalar(1));
            // Gamma.fill(std::min(1.0, 0.5 * tau));
        }

        // Set primal variable based on duals
        set_primal();
    }

    inline void set_seed(Index seed) { m_rng.seed(seed); }

    inline Index solve_vanilla(
        std::vector<Scalar>& dual_objfns, std::vector<Scalar>& primal_objfns,
        Index max_iter, Scalar tol,
        Index verbose = 0, Index trace_freq = 100,
        std::ostream& cout = std::cout)
    {
        // Main iterations
        Index i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta();
            update_Lambda_beta();
            update_Gamma_beta();

            // Compute difference of xi and beta
            const Scalar xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : Scalar(0);
            const Scalar beta_diff = (m_beta - old_beta).norm();

            // Print progress
            if (verbose && (i % trace_freq == 0))
            {
                Scalar dual = dual_objfn();
                dual_objfns.push_back(dual);
                Scalar primal = primal_objfn();
                primal_objfns.push_back(primal);
                cout << "Iter " << i << ", dual_objfn = " << dual <<
                    ", primal_objfn = " << primal <<
                    ", xi_diff = " << xi_diff <<
                    ", beta_diff = " << beta_diff << std::endl;
            }

            // Convergence test based on change of variable values
            const bool vars_conv = (xi_diff < tol) && (beta_diff < tol);
            if (vars_conv)
                break;
        }

        return i;
    }

    inline Index solve(
        std::vector<Scalar>& dual_objfns, std::vector<Scalar>& primal_objfns,
        Index max_iter, Scalar tol,
        Index verbose = 0, Index trace_freq = 100,
        std::ostream& cout = std::cout)
    {
        // Free variable sets
        internal::reset_fv_set(m_fv_feas, m_K);
        internal::reset_fv_set(m_fv_relu, m_L, m_n);
        internal::reset_fv_set(m_fv_rehu, m_H, m_n);

        // Shrinking thresholds
        constexpr Scalar Inf = std::numeric_limits<Scalar>::infinity();
        Scalar xi_min_pg = Inf, lambda_min_pg = Inf, gamma_min_pg = Inf;
        Scalar xi_max_pg = -Inf, lambda_max_pg = -Inf, gamma_max_pg = -Inf;

        // Main iterations
        Index i = 0;
        Vector old_xi(m_K), old_beta(m_d);
        for(; i < max_iter; i++)
        {
            old_xi.noalias() = m_xi;
            old_beta.noalias() = m_beta;

            update_xi_beta(m_fv_feas, xi_min_pg, xi_max_pg);
            update_Lambda_beta(m_fv_relu, lambda_min_pg, lambda_max_pg);
            update_Gamma_beta(m_fv_rehu, gamma_min_pg, gamma_max_pg);

            // Compute difference of xi and beta
            const Scalar xi_diff = (m_K > 0) ? (m_xi - old_xi).norm() : Scalar(0);
            const Scalar beta_diff = (m_beta - old_beta).norm();

            // Convergence test based on change of variable values
            const bool vars_conv = (xi_diff < tol) && (beta_diff < tol);
            // Convergence test based on PG
            const bool pg_conv = (xi_max_pg - xi_min_pg < tol) &&
                                 (std::abs(xi_max_pg) < tol) &&
                                 (std::abs(xi_min_pg) < tol) &&
                                 (lambda_max_pg - lambda_min_pg < tol) &&
                                 (std::abs(lambda_max_pg) < tol) &&
                                 (std::abs(lambda_min_pg) < tol) &&
                                 (gamma_max_pg - gamma_min_pg < tol) &&
                                 (std::abs(gamma_max_pg) < tol) &&
                                 (std::abs(gamma_min_pg) < tol);
            // Whether we are using all variables
            const bool all_vars = (m_fv_feas.size() == static_cast<std::size_t>(m_K)) &&
                                  (m_fv_relu.size() == static_cast<std::size_t>(m_L * m_n)) &&
                                  (m_fv_rehu.size() == static_cast<std::size_t>(m_H * m_n));

            // Print progress
            if (verbose && (i % trace_freq == 0))
            {
                Scalar dual = dual_objfn();
                dual_objfns.push_back(dual);
                Scalar primal = primal_objfn();
                primal_objfns.push_back(primal);
                cout << "Iter " << i << ", dual_objfn = " << dual <<
                    ", primal_objfn = " << primal <<
                    ", xi_diff = " << xi_diff <<
                    ", beta_diff = " << beta_diff << std::endl;
                if (verbose >= 2)
                {
                    cout << "    xi (" << m_fv_feas.size() << "/" << m_K <<
                        "), lambda (" << m_fv_relu.size() << "/" << m_L * m_n <<
                        "), gamma (" << m_fv_rehu.size() << "/" << m_H * m_n << ")" << std::endl;
                    cout << "    xi_pg = (" << xi_min_pg << ", " << xi_max_pg <<
                        "), lambda_pg = (" << lambda_min_pg << ", " << lambda_max_pg <<
                        "), gamma_pg = (" << gamma_min_pg << ", " << gamma_max_pg << ")" << std::endl;
                }
            }

            // If variable value or PG converges but not on all variables,
            // use all variables in the next iteration
            if ((vars_conv || pg_conv) && (!all_vars))
            {
                if (verbose)
                {
                    cout << "*** Iter " << i <<
                        ", free variables converge; next test on all variables" << std::endl;
                }
                internal::reset_fv_set(m_fv_feas, m_K);
                internal::reset_fv_set(m_fv_relu, m_L, m_n);
                internal::reset_fv_set(m_fv_rehu, m_H, m_n);
                xi_min_pg = lambda_min_pg = gamma_min_pg = Inf;
                xi_max_pg = lambda_max_pg = gamma_max_pg = -Inf;
                // Also recompute beta to improve precision
                // set_primal();
                continue;
            }
            if (all_vars && (vars_conv || pg_conv))
                break;
        }

        return i;
    }

    Vector& get_beta_ref() { return m_beta; }
    Vector& get_xi_ref() { return m_xi; }
    Matrix& get_Lambda_ref() { return m_Lambda; }
    Matrix& get_Gamma_ref() { return m_Gamma; }
};

// Main solver interface
// template <typename Matrix = Eigen::MatrixXd, typename Index = int>
template <typename DerivedMat, typename DerivedVec, typename Index = int>
void rehline_solver(
    ReHLineResult<typename DerivedMat::PlainObject, Index>& result,
    const Eigen::MatrixBase<DerivedMat>& X, const Eigen::MatrixBase<DerivedMat>& A,
    const Eigen::MatrixBase<DerivedVec>& b,
    const Eigen::MatrixBase<DerivedMat>& U, const Eigen::MatrixBase<DerivedMat>& V,
    const Eigen::MatrixBase<DerivedMat>& S, const Eigen::MatrixBase<DerivedMat>& T, const Eigen::MatrixBase<DerivedMat>& Tau,
    Index max_iter, typename DerivedMat::Scalar tol, Index shrink = 1,
    Index verbose = 0, Index trace_freq = 100,
    std::ostream& cout = std::cout
)
{
    // Create solver
    ReHLineSolver<typename DerivedMat::PlainObject, Index> solver(X, U, V, S, T, Tau, A, b);

    // Initialize parameters
    solver.init_params();

    // Main iterations
    std::vector<typename DerivedMat::Scalar> dual_objfns;
    std::vector<typename DerivedMat::Scalar> primal_objfns;
    Index niter;
    if (shrink > 0)
    {
        solver.set_seed(shrink);
        niter = solver.solve(dual_objfns, primal_objfns, max_iter, tol, verbose, trace_freq, cout);
    } else {
        niter = solver.solve_vanilla(dual_objfns, primal_objfns, max_iter, tol, verbose, trace_freq, cout);
    }

    // Save result
    result.beta.swap(solver.get_beta_ref());
    result.xi.swap(solver.get_xi_ref());
    result.Lambda.swap(solver.get_Lambda_ref());
    result.Gamma.swap(solver.get_Gamma_ref());
    result.niter = niter;
    result.dual_objfns.swap(dual_objfns);
    result.primal_objfns.swap(primal_objfns);
}

// Main SVM solver interface
// template <typename Matrix = Eigen::MatrixXd, typename Index = int>
template <typename DerivedMat, typename DerivedVec, typename Index = int>
void rehline_svm(
    ReHLineResult<typename DerivedMat::PlainObject, Index>& result,
    const Eigen::MatrixBase<DerivedMat>& X,
    const Eigen::MatrixBase<DerivedVec>& y,
    typename DerivedMat::Scalar C,
    Index max_iter, typename DerivedMat::Scalar tol, Index shrink = 1,
    Index verbose = 0, Index trace_freq = 100,
    std::ostream& cout = std::cout
)
{
    using Matrix = typename DerivedMat::PlainObject;
    using Vector = typename DerivedVec::PlainObject;
    const Index n = X.rows();
    const Index d = X.cols();

    Matrix U = -C / n * y.transpose();
    Matrix V = Matrix::Constant(1, n, C / n);
    Matrix A(0, d);
    Vector b(0);
    Matrix S(0, n), T(0, n), Tau(0, n);

    rehline_solver(result, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq, cout);
}


}  // namespace rehline


#endif  // REHLINE_H
