# nbzinbAmoah

<!-- badges: start -->
[![R-CMD-check](https://github.com/AbigailAmoah/nbzinbAmoah/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/AbigailAmoah/nbzinbAmoah/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

An R package for **Negative Binomial (NB)** and **Zero-Inflated Negative Binomial (ZINB)** regression, built from scratch using iteratively reweighted least squares (IRLS) and an EM algorithm — without any general-purpose optimizer.

## Installation

```r
# Install from GitHub
devtools::install_github("AbigailAmoah/nbzinbAmoah")
```

## Usage

### Negative Binomial Regression

```r
library(nbzinbAmoah)
data("bioChemists", package = "pscl")

fit_nb <- nbreg(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)
summary(fit_nb)
```

### Zero-Inflated Negative Binomial Regression

```r
fit_zinb <- zinbreg(
  art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment,
  data = bioChemists
)
summary(fit_zinb)
```

## Methods implemented

| Function | Algorithm |
|---|---|
| `nbreg()` | IRLS for β + Newton–Raphson for θ |
| `zinbreg()` | EM algorithm; M-steps are IRLS |

Both functions return S3 objects with `print`, `summary`, `coef`, `vcov`, `logLik`, `fitted`, `residuals`, `nobs`, and `predict` methods.

## Notes on the implementation

The file `CLAUDE.md` lists the project constraints. The prohibited functions (`optim`, `glm`, `lm`, etc.) are **never** called inside the fitting code in `R/`. They appear only in `tests/` and `inst/benchmark/` for benchmarking purposes.

## Pkgdown website

[https://AbigailAmoah.github.io/nbzinbAmoah/](https://AbigailAmoah.github.io/nbzinbAmoah/)
