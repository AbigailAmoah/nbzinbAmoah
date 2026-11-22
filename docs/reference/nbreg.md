# Negative Binomial Regression

Fits a negative binomial (NB2) regression model using iteratively
reweighted least squares (IRLS) for the coefficient vector beta,
alternated with Newton-Raphson updates for the dispersion parameter
theta. No general-purpose optimizer is used.

## Usage

``` r
nbreg(formula, data, offset = NULL, control = nbreg_control(), weights = NULL)
```

## Arguments

- formula:

  A formula of the form `y ~ x1 + x2 + ...`.

- data:

  A data frame containing the variables in the formula.

- offset:

  An optional numeric vector of offsets on the log-mean scale. Can also
  be specified in the formula via
  [`offset()`](https://rdrr.io/r/stats/offset.html).

- control:

  A list of control parameters from
  [`nbreg_control`](https://AbigailAmoah.github.io/nbzinbAmoah/reference/nbreg_control.md).

- weights:

  Optional prior weights (all 1 by default).

## Value

An object of class `"nbreg"` with components:

- coefficients:

  Named vector of estimated beta coefficients.

- theta:

  Estimated dispersion parameter theta \> 0.

- SE_theta:

  Standard error of log(theta).

- loglik:

  Maximised log-likelihood.

- vcov:

  Variance-covariance matrix of beta (treating theta as known).

- fitted.values:

  Fitted means mu_i.

- residuals:

  Response residuals y - mu.

- weights:

  Prior weights used.

- nobs:

  Number of observations.

- df.residual:

  Residual degrees of freedom.

- iter:

  Number of outer iterations.

- converged:

  Logical; TRUE if converged within maxit.

- call:

  The matched call.

- terms:

  The terms object.

- model:

  The model frame.

## Warnings

A warning is issued if the algorithm reaches `maxit` iterations without
converging, or if \\\hat heta\\ exceeds \\10^4\\, which means the data
show essentially no overdispersion (the Poisson limit).

## Examples

``` r
if (requireNamespace("pscl", quietly = TRUE)) {
  data("bioChemists", package = "pscl")
  fit <- nbreg(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)
  summary(fit)
}
#> 
#> Negative Binomial Regression
#> Call: nbreg(formula = art ~ fem + mar + kid5 + phd + ment, data = bioChemists)
#> 
#> Coefficients:
#>              Estimate Std. Error z value Pr(>|z|)    
#> (Intercept)  0.256144   0.137348   1.865 0.062191 .  
#> femWomen    -0.216418   0.072636  -2.979 0.002887 ** 
#> marMarried   0.150489   0.082097   1.833 0.066791 .  
#> kid5        -0.176415   0.052813  -3.340 0.000837 ***
#> phd          0.015271   0.035873   0.426 0.670326    
#> ment         0.029082   0.003214   9.048  < 2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Theta: 2.2644  (SE of log-theta: 0.1198)
#> Log-likelihood: -1560.9583 on 908 Df
#> n = 915  Converged: TRUE  Iterations: 4
```
