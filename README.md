## ReHLine-SVM

**ReHLine-SVM** is a tiny and header-only C++ library aiming to be
the **fastest** linear SVM solver. The whole library is a single
header file [rehline.h](rehline.h), and its only dependency is the
[Eigen](https://eigen.tuxfamily.org) library, which is also header-only.

**ReHLine-SVM** solves the following optimization problem:

$$
  \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n ( 1 - y_i \mathbf{\beta}^\intercal \mathbf{x}_ i )_+ + \frac{1}{2} \Vert \mathbf{\beta} \Vert_2^2,
$$

where $\mathbf{x}_ i \in \mathbb{R}^d$ is a feature vector, $y_i \in \\{-1, 1\\}$ is a binary label, and $(x)_+=\max\\{x,0\\}$.

### Quick Start

The `example1.cpp` file shows the basic use of the **ReHLine-SVM**
library. The key step is to first define a result variable `res`,
and then call the `rehline::rehline_svm()` function:

```cpp
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
```

The variable `X` is a matrix of dimension $n\times d$, and `y`
is a vector of length $n$ taking values of 1 or -1.
For better performance, `X` is suggested to be a row-majored matrix
in Eigen, whose type can be defined as follows:

```cpp
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
```

The complete code can be found in the [example1.cpp](example1.cpp) file.

Assuming the Eigen library has been extracted to the current directory,
we can use the following command to compile the program:

```bash
g++ -std=c++11 -O2 -I. example1.cpp -o example1
```

Running `example1` gives the following possible output:

```
niter = 100
beta =
-0.284341
 0.680663
-0.273076
-0.134658
-0.165271
0.0601155
0.0697878
 0.150636
 0.343303
 0.226581
```

### Benchmark

It is widely recognized that the celebrated
[Liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) library
is one of the fastest linear SVM solvers,
and we seriously take the challenge.

We include the latest version of Liblinear (2.47 for now) in the
`liblinear` directory, and slightly modify its main program file
`train.c` to include the computing time
(the [train2.c](liblinear/train2.c) file).

Then we compare Liblinear and **ReHLine-SVM** on a large data set with
5,000,000 observations and 18 features. To reproduce the experiment,
download the [SUSY](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz) data,
and extract it into the `data` directory.

The following command is used to compile the Liblinear solver:

```bash
gcc -O3 -fPIC -c liblinear/blas/daxpy.c
gcc -O3 -fPIC -c liblinear/blas/ddot.c
gcc -O3 -fPIC -c liblinear/blas/dnrm2.c
gcc -O3 -fPIC -c liblinear/blas/dscal.c
ar rcs blas.a daxpy.o ddot.o dnrm2.o dscal.o
g++ -std=c++11 -O2 -fPIC -DNDEBUG -c liblinear/newton.cpp
g++ -std=c++11 -O2 -fPIC -DNDEBUG -c liblinear/linear.cpp
gcc -O3 -fPIC -c liblinear/train2.c
g++ -std=c++11 -O2 train2.o newton.o linear.o blas.a -o run_liblinear
```

And then run the program:

```bash
./run_liblinear -s 3 -c 2e-5 -e 1e-5 data/SUSY
```

It will generate a model file named `SUSY.model`, and the output
shows that its model solving takes 11.58 seconds.

```
......**
optimization finished, #iter = 64
Objective value = -58.018530
nSV = 3122272
Computation time: 11.583774 seconds.
```

Then we use **ReHLine-SVM** to compute it
(the complete code is in [example2.cpp](example2.cpp)),
and compare its result with
that of Liblinear:

```bash
g++ -std=c++11 -O2 -I. -DNDEBUG example2.cpp -o run_rehline
./run_rehline data/SUSY SUSY.model
```

The output shows that **ReHLine-SVM** only takes about 4.7 seconds
while achieving a smaller objective function value (ReHline-SVM 58.072/Liblinear 58.1859).

```
Iter 0, dual_objfn = -45.7806, primal_objfn = 66.6884, xi_diff = 0, beta_diff = 21.8619
*** Iter 20, free variables converge; next test on all variables
*** Iter 23, free variables converge; next test on all variables
Computation time: 4.71375 seconds

beta =
    0.942592
 0.000327085
-0.000597487
   -0.124389
 0.000449019
 0.000219511
     2.06514
  0.00175777
    0.217093
  -0.0921886
   -0.560916
    0.695706
    -1.25522
   -0.304249
   -0.579766
   -0.135974
   -0.907076
           0
objfn = 58.072

beta(liblinear) =
    0.907424
 0.000313113
-0.000610905
  -0.0973855
 0.000380076
 0.000225803
     2.04279
  0.00178176
    0.229943
  -0.0821463
   -0.560707
    0.669993
    -1.31528
   -0.297069
   -0.569391
   -0.118391
   -0.896777
    0.252936
objfn(liblinear) = 58.1859
```

### Library Use

The core of this library is the function

```cpp
rehline::rehline_svm(result, X, y, C, max_iter, tol, shrink, verbose, trace_freq, cout)
```

The meaning of each argument is as follows:

- `result`: (output) an object containing the optimization results.
- `X`: (input) an $n\times d$ data matrix, preferred to be row-majored.
- `y`: (input) a response vector of length $n$ taking values of 1 or -1.
- `C`: (input) a scalar standing for the cost parameter.
- `max_iter`: (input) an integer representing the maximum number of iterations.
- `tol`: (input) a scalar giving the convergence tolerance.
- `shrink`: (input) if it is a positive integer, then a shrinking scheme is used to accelerate the algorithm, and the value of this argument is used as a random seed; otherwise, the vanilla algorithm is used.
- `verbose`: (input) level of verbosity, taking values of 0, 1, or 2.
- `trace_freq` (input) trace objective function values every `trace_freq` iterations; only works if `verbose > 0`.
- `cout`: the output stream object, default to be `std::cout`.

### Extensions

The **ReHLine-SVM** library is based on the
[ReHLine](https://rehline.github.io/) algorithm and solver, which takes SVM
as a special case. ReHLine can do much more, including the smoothed SVM,
Huber regression, quantile regression, etc. Please see the
[ReHLine project page](https://rehline.github.io/) for details.

### License

**ReHLine-SVM** is open source under the MIT license.

### Citation

Please consider to cite our article if you find **ReHLine-SVM** or
the ReHLine algorithm/solver useful.

```
@inproceedings{daiqiu2023rehline,
    title={ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence},
    author={Dai, Ben and Yixuan Qiu},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023}
}
```
