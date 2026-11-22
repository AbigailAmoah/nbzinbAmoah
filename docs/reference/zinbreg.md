# Zero-Inflated Negative Binomial Regression

Fits a zero-inflated negative binomial (ZINB) regression via an EM
algorithm whose M-steps are iteratively reweighted least squares (IRLS).
No general-purpose optimizer is used.

## Usage

``` r
zinbreg(
  formula,
  data,
  offset = NULL,
  control = zinbreg_control(),
  weights = NULL
)
```

## Arguments

- formula:

  A two-part formula `y ~ count_covariates | zero_covariates`. If `|` is
  omitted, the same covariates are used in both parts.

- data:

  A data frame.

- offset:

  An optional offset for the count part (log scale).

- control:

  A list from
  [`zinbreg_control`](https://AbigailAmoah.github.io/nbzinbAmoah/reference/zinbreg_control.md).

- weights:

  Optional prior weights.

## Value

An object of class `"zinbreg"` with components:

- coefficients:

  List with `count` (beta) and `zero` (gamma).

- theta:

  Estimated dispersion parameter.

- loglik:

  Observed-data log-likelihood at convergence.

- vcov:

  Variance-covariance matrix (numerical Hessian of observed loglik).

- fitted.values:

  Expected counts \\(1-\pi_i)\mu_i\\.

- fitted.pi:

  Estimated zero-inflation probabilities \\\pi_i\\.

- fitted.mu:

  Estimated NB means \\\mu_i\\.

- residuals:

  Response residuals \\y - (1-\pi)\mu\\.

- nobs:

  Number of observations.

- df.residual:

  Residual degrees of freedom.

- iter:

  Number of EM iterations.

- converged:

  Logical.

- call:

  The matched call.

## Details

The model has two components:

- **Count part**: \\\log \mu_i = x_i^\top \beta + o_i\\

- **Zero part**: \\\text{logit}\\\pi_i = z_i^\top \gamma\\

Specify both parts with a two-part formula `y ~ x1 + x2 | z1 + z2`. If
no `|` is present, the same covariates are used for both parts.

## Warnings

A warning is issued if the EM algorithm reaches `maxit` iterations
without converging, if the observed-data log-likelihood ever decreases,
or if \\\hat heta\\ exceeds \\10^4\\, which means the count part shows
essentially no overdispersion (the Poisson limit).

## Examples

``` r
if (requireNamespace("pscl", quietly = TRUE)) {
  data("bioChemists", package = "pscl")
  fit <- zinbreg(art ~ fem + mar + kid5 + phd + ment |
                       fem + mar + kid5 + phd + ment,
                 data = bioChemists)
  summary(fit)
}
#> 
#> Zero-Inflated Negative Binomial Regression
#> Call: zinbreg(formula = art ~ fem + mar + kid5 + phd + ment | fem + 
#>     mar + kid5 + phd + ment, data = bioChemists)
#> 
#> 
#> Count model coefficients (NB with log link):
#>               Estimate Std. Error z value Pr(>|z|)    
#> (Intercept)  0.4167489  0.1435963   2.902  0.00371 ** 
#> femWomen    -0.1955045  0.0755924  -2.586  0.00970 ** 
#> marMarried   0.0975840  0.0844519   1.155  0.24789    
#> kid5        -0.1517307  0.0542061  -2.799  0.00512 ** 
#> phd         -0.0006995  0.0362696  -0.019  0.98461    
#> ment         0.0247859  0.0034924   7.097 1.27e-12 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Log(theta): 0.9764  (SE: 0.1355)  =>  theta = 2.6548
#> 
#> Zero-inflation model coefficients (binomial with logit link):
#>             Estimate Std. Error z value Pr(>|z|)   
#> (Intercept) -0.19184    1.32272  -0.145  0.88468   
#> femWomen     0.63595    0.84885   0.749  0.45375   
#> marMarried  -1.49931    0.93859  -1.597  0.11017   
#> kid5         0.62841    0.44276   1.419  0.15581   
#> phd         -0.03768    0.30799  -0.122  0.90262   
#> ment        -0.88222    0.31619  -2.790  0.00527 **
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Log-likelihood: -1549.9909 on 902 Df
#> n = 915  Converged: TRUE  Iterations: 111
```
