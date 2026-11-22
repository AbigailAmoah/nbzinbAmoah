# Control parameters for zinbreg

Control parameters for zinbreg

## Usage

``` r
zinbreg_control(
  maxit = 300,
  irls_maxit = 50,
  theta_maxit = 50,
  epsilon = 1e-08,
  trace = FALSE
)
```

## Arguments

- maxit:

  Maximum EM iterations (default 300).

- irls_maxit:

  Max IRLS iterations within each M-step (default 50).

- theta_maxit:

  Max Newton-Raphson iterations for theta (default 50).

- epsilon:

  Convergence tolerance on observed log-likelihood (default 1e-8).

- trace:

  Logical; if TRUE print log-likelihood at each EM iteration.

## Value

A list of control parameters.
