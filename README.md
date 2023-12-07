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
niter = 118
beta =
  0.588211
 0.0826256
 0.0709005
  0.339442
 -0.414267
 0.0141788
  0.525987
 -0.176368
-0.0932384
  0.282286
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
g++ -std=c++11 -O2 -fPIC -c liblinear/newton.cpp
g++ -std=c++11 -O2 -fPIC -c liblinear/linear.cpp
gcc -O3 -fPIC -c liblinear/train2.c
g++ -std=c++11 -O2 train2.o newton.o linear.o blas.a -o liblinear
```

And then run the program:

```bash
liblinear -s 3 -c 2e-5 -e 0.001 data/SUSY
```

It will generate a model file named `SUSY.model`, and the output
shows that its model solving takes 132.875 seconds.

```................*..*.*
optimization finished, #iter = 192
Objective value = -58.018530
nSV = 3122349
Computation time: 132.875000 seconds.
```

Then we use **ReHLine-SVM** to compute it
(the complete code is in [example2.cpp](example2.cpp)),
and compare its result with
that of Liblinear:

```bash
g++ -std=c++11 -O2 -I. example2.cpp -o rehline
rehline data/SUSY SUSY.model
```

The output shows that **ReHLine-SVM** only takes about 30 seconds
while achieving a smaller objective function value (ReHline-SVM 58.072/Liblinear 58.1859).

```
Iter 0, dual_objfn = -45.7976, primal_objfn = 66.6771, xi_diff = 0, beta_diff = 21.8617
Iter 100, dual_objfn = -57.9117, primal_objfn = 58.1417, xi_diff = 0, beta_diff = 0.00384903
*** Iter 115, free variables converge; next test on all variables
*** Iter 136, free variables converge; next test on all variables
*** Iter 138, free variables converge; next test on all variables
Computation time: 29.8851 seconds

beta =
     0.94276
 0.000308242
-0.000653667
   -0.124244
 0.000451755
 0.000165797
     2.06511
  0.00179583
    0.216882
  -0.0914909
   -0.560838
    0.695612
    -1.25535
   -0.304982
   -0.579794
   -0.136381
   -0.906826
           0
objfn = 58.072

beta(liblinear) =
    0.907439
 0.000313963
-0.000614302
  -0.0973794
 0.000376164
 0.000217329
      2.0428
  0.00178343
    0.229951
  -0.0821376
   -0.560714
    0.669999
    -1.31528
   -0.297058
     -0.5694
   -0.118384
   -0.896794
    0.252937
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
- `y`: (input) a length-$n$ response vector taking values of 1 or -1.
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
