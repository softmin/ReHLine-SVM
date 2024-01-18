#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include "rehline.h"

using Scalar = double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Get the dimensions of the data matrix by scanning the data file
void get_dim(const std::string& filename, std::size_t& nrow, std::size_t& ncol)
{
    nrow = 0;
    ncol = 0;

    // Open file
    std::ifstream file(filename);
    if (file.is_open())
    {
        // Read each line
        std::string line;
        while (std::getline(file, line))
        {
            // Find the last colon
            std::size_t colon = line.rfind(':');
            // Find the space before the last colon
            std::size_t space = line.rfind(' ', colon);
            std::size_t col = std::stoi(line.substr(space + 1, colon - space - 1));
            ncol = std::max(ncol, col);
            nrow++;
        }
        // std::cout << nrow << std::endl;
        // std::cout << ncol << std::endl;
        file.close();
    }
}

// Read in data
void read_data(
    const std::string& filename, std::size_t nrow, std::size_t ncol,
    Matrix& x, Vector& y
)
{
    x.resize(nrow, ncol);
    x.setZero();
    y.resize(nrow);

    // Open file
    std::ifstream file(filename);
    if (file.is_open())
    {
        // Read each line
        std::string line;
        std::size_t i = 0;
        while (std::getline(file, line) && i < nrow)
        {
            // Read the response value
            std::size_t space = line.find(' ');
            int label = std::stoi(line.substr(0, space));
            y[i] = (label < 0.5) ? -1.0 : 1.0;
            // Read features
            std::size_t end = line.find(' ', space + 1);
            while (end != std::string::npos)
            {
                // line.substr(space + 1, end - space - 1) contains "<col>:<val>"
                // std::cout << line.substr(space + 1, end - space - 1) << std::endl;
                std::size_t colon = line.find(':', space + 1);
                if (colon != std::string::npos)
                {
                    int j = std::stoi(line.substr(space + 1, colon - space - 1));
                    Scalar val = std::stod(line.substr(colon + 1, end - colon - 1));
                    // std::cout << "j = " << j << ", val = " << val << std::endl;
                    x(i, j - 1) = val;
                }
                space = end;
                end = line.find(' ', space + 1);
            }
            // Explicitly read the last feature
            std::size_t colon = line.find(':', space + 1);
            if (colon != std::string::npos)
            {
                int j = std::stoi(line.substr(space + 1, colon - space - 1));
                Scalar val = std::stod(line.substr(colon + 1, std::string::npos));
                x(i, j - 1) = val;
            }

            i++;
        }
        file.close();
    }
}

// Read Liblinear model file
void read_model(const std::string& filename, Vector& beta)
{
    // Open file
    std::ifstream file(filename);
    if (file.is_open())
    {
        // Find the line containing the string "w"
        std::string line;
        while (std::getline(file, line) && (line != "w"))
        {
        }
        // Read the remaining lines
        std::vector<Scalar> betas;
        while (std::getline(file, line))
        {
            betas.push_back(std::stod(line));
        }
        // Copy the data to beta
        beta.resize(betas.size());
        std::copy(betas.begin(), betas.end(), beta.data());
        file.close();
    }
}

int main(int argc, char *argv[])
{
    // The second argument gives the data file name
    if (argc < 2)
    {
        std::cout << "Please specify data file." << std::endl;
        return 1;
    }
    const std::string filename = std::string(argv[1]);
    std::cout << "*** Use data file \"" << filename << "\" ***" << std::endl << std::endl;

    // Get data dimensions
    std::cout << "*** Determining data dimension... ***" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::size_t n = 0, p = 0;
    get_dim(filename, n, p);
    std::cout << "nrow = " << n << ", ncol = " << p << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "*** Finished in " << elapsed.count() << " seconds. ***" << std::endl << std::endl;
    if (n < 1 || p < 1)
    {
        std::cout << "Reading data failed." << std::endl;
        return 1;
    }

    // Read in data
    std::cout << "*** Reading data... ***" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Matrix X(n, p);
    Vector y(n);
    read_data(filename, n, p, X, y);
    std::cout << "X =\n" << X.topRows(6) << std::endl << std::endl;
    std::cout << "y = " << y.head(6).transpose() << std::endl;
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "*** Finished in " << elapsed.count() << " seconds. ***" << std::endl << std::endl;

    // Setting parameters
    Scalar C = 100.0;
    int max_iter = 10000;
    Scalar tol = 1e-5;
    int shrink = 1;
    int verbose = 1;
    int trace_freq = 100;

    // Run the solver
    start = std::chrono::high_resolution_clock::now();
    rehline::ReHLineResult<Matrix> res;
    rehline::rehline_svm(res, X, y, C, max_iter, tol, shrink, verbose, trace_freq);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Computation time: " << elapsed.count() << " seconds" << std::endl << std::endl;

    // Print estimated beta and the corresponding objective function value
    std::cout << "beta =\n" << res.beta << std::endl;
    Scalar objfn = C / n * (1.0 - (X * res.beta).array() * y.array()).max(0.0).sum() +
        0.5 * res.beta.squaredNorm();
    std::cout << "objfn = " << objfn << std::endl << std::endl;

    // If a Liblinear model file has been supplied, compare the results
    if (argc >= 3)
    {
        Vector liblinear_beta;
        read_model(argv[2], liblinear_beta);
        std::cout << "beta(liblinear) =\n" << liblinear_beta << std::endl;
        Scalar liblinear_objfn = C / n * (1.0 - (X * liblinear_beta).array() * y.array()).max(0.0).sum() +
            0.5 * liblinear_beta.squaredNorm();
        std::cout << "objfn(liblinear) = " << liblinear_objfn << std::endl;
    }

    return 0;
}
