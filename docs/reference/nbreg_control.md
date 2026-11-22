# Control parameters for nbreg

Control parameters for nbreg

## Usage

``` r
nbreg_control(maxit = 200, theta_maxit = 50, epsilon = 1e-08, trace = FALSE)
```

## Arguments

- maxit:

  Maximum number of outer IRLS iterations (default 200).

- theta_maxit:

  Maximum Newton-Raphson iterations for theta (default 50).

- epsilon:

  Convergence tolerance on log-likelihood change (default 1e-8).

- trace:

  Logical; if TRUE print log-likelihood at each iteration.

## Value

A list of control parameters.
