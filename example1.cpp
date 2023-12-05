#include <iostream>
#include <Eigen/Core>
#include "rehline.h"

int main()
{
    // The preferred matrix type is row-majored
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector = Eigen::VectorXd;

    // Dimensions
    const int n = 1000;
    const int p = 10;
    
    // Simulate data
    std::srand(123);
    Matrix X = Matrix::Random(n, p);
    Vector y(n);
    for (int i = 0; i < n; i++)
    {
        if (std::rand() % 2 == 0)
            y[i] = 1;
        else
            y[i] = -1;
    }

    // Setting parameters
    double C = 100.0;
    int max_iter = 1000;
    double tol = 1e-5;
    int shrink = 1;
    int verbose = 0;
    int trace_freq = 100;

    // Run the solver
    rehline::ReHLineResult<Matrix> res;
    rehline::rehline_svm(res, X, y, C, max_iter, tol, shrink, verbose, trace_freq);

    // Print the estimated beta
    std::cout << "niter = " << res.niter << "\nbeta =\n" << res.beta << std::endl;

    return 0;
}
