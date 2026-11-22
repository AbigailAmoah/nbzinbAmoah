# 1 Introduction

# 1 Introduction

Count data arise frequently in applied statistics: the number of
physician office visits, the number of articles published, or the number
of insurance claims. The simplest model for count data is the Poisson
regression, which assumes that the mean equals the variance. In
practice, two violations are common:

**Overdispersion.** The variance often exceeds the mean. The negative
binomial (NB) model addresses this by introducing a dispersion parameter
$`\theta > 0`$ so that $`\text{Var}(Y_i) = \mu_i + \mu_i^2/\theta`$. As
$`\theta \to \infty`$ the model reduces to the Poisson.

**Excess zeros.** Many count datasets contain more zeros than a Poisson
(or even NB) distribution predicts. The zero-inflated negative binomial
(ZINB) model ([Lambert 1992](#ref-lambert1992)) handles this through a
two-component mixture: with probability $`\pi_i`$ the observation is a
structural zero (e.g., a person who never visits a doctor); with
probability $`1-\pi_i`$ it comes from an NB distribution, which can
itself produce zeros.

This report documents the **nbzinbAmoah** R package, which fits both
models from scratch using iteratively reweighted least squares (IRLS)
([McCullagh and Nelder 1989](#ref-mccullagh1989)) and an
expectation-maximisation (EM) algorithm — without any general-purpose
optimizer. We benchmark our implementation against the reference
functions `MASS::glm.nb()` ([Venables and Ripley
2002](#ref-venables2002)) and `pscl::zeroinfl()` ([Zeileis et al.
2008](#ref-zeileis2008)), and apply the package to the NMES1988 dataset
from the **AER** package ([Kleiber and Zeileis 2008](#ref-kleiber2008)).

# 2 Methods

## 2.1 Negative Binomial Regression

Let $`Y_i`$ be a count and $`x_i`$ a $`p`$-vector of covariates,
$`i = 1, \ldots, n`$. The NB2 model assumes
``` math
P(Y_i = y) = \frac{\Gamma(y+\theta)}{\Gamma(\theta)\,y!}
  \left(\frac{\theta}{\theta+\mu_i}\right)^\theta
  \left(\frac{\mu_i}{\theta+\mu_i}\right)^y, \quad y = 0,1,2,\ldots
```
with log link $`\log\mu_i = \eta_i = x_i^\top\beta + o_i`$, where
$`o_i`$ is an optional offset. Then $`E(Y_i) = \mu_i`$ and
$`\text{Var}(Y_i) = \mu_i + \mu_i^2/\theta`$. With prior weights $`a_i`$
($`a_i = 1`$ for an ordinary fit) the log-likelihood is
``` math
\ell(\beta,\theta) = \sum_{i=1}^n a_i\Bigl[\log\Gamma(y_i+\theta) - \log\Gamma(\theta)
  - \log y_i! + \theta\log\theta + y_i\log\mu_i - (\theta+y_i)\log(\theta+\mu_i)\Bigr].
```
This is the parameterisation used by `MASS::glm.nb()`, whose `theta` is
the same $`\theta`$.

**IRLS for $`\beta`$ (with $`\theta`$ fixed).** For a GLM with log link
and variance $`V(\mu) = \mu + \mu^2/\theta`$, Fisher scoring is
equivalent to repeated weighted least squares with working weights and
working response
``` math
w_i = a_i\,\frac{(\partial\mu_i/\partial\eta_i)^2}{V(\mu_i)}
    = a_i\,\frac{\theta\,\mu_i}{\theta + \mu_i}, \qquad
z_i = (\eta_i - o_i) + \frac{y_i - \mu_i}{\mu_i},
```
``` math
\beta^{(t+1)} = (X^\top W X)^{-1} X^\top W z, \qquad W = \text{diag}(w_1,\ldots,w_n).
```
The weighted least-squares step is computed with `solve()`; `lm()`,
`lm.wfit()` and `glm.fit()` are not used.

**Newton–Raphson for $`\theta`$ (with $`\beta`$ fixed).** The score and
second derivative of $`\ell`$ with respect to $`\theta`$ are
``` math
U(\theta) = \sum_i a_i\Bigl[\psi(y_i+\theta) - \psi(\theta) + \log\theta + 1
  - \log(\theta+\mu_i) - \frac{y_i+\theta}{\theta+\mu_i}\Bigr],
```
``` math
H(\theta) = \sum_i a_i\Bigl[\psi'(y_i+\theta) - \psi'(\theta) + \frac{1}{\theta}
  - \frac{2}{\theta+\mu_i} + \frac{y_i+\theta}{(\theta+\mu_i)^2}\Bigr],
```
where $`\psi`$ and $`\psi'`$ are `digamma()` and `trigamma()`. We update
$`\phi = \log\theta`$, which keeps $`\theta > 0`$. By the chain rule
$`U_\phi = \theta U`$ and $`H_\phi = \theta^2 H + \theta U`$, and the
update is $`\phi \leftarrow \phi - U_\phi / H_\phi`$.

**Standard errors.** At convergence,
$`\widehat{\text{Var}}(\hat\beta) = (X^\top\hat W X)^{-1}`$, treating
$`\hat\theta`$ as known as `glm.nb()` does, and
$`\text{SE}(\hat\theta) = \{-H(\hat\theta)\}^{-1/2}`$. The package
stores $`\text{SE}(\log\hat\theta) = (-H_\phi)^{-1/2}`$. Because
$`U(\hat\theta) = 0`$ at the MLE, $`H_\phi = \hat\theta^2 H`$, so
$`\text{SE}(\hat\theta) = \hat\theta\,\text{SE}(\log\hat\theta)`$.

## 2.2 Zero-Inflated Negative Binomial Regression

The ZINB model is a two-component mixture. Let $`D_i = 1`$ if
observation $`i`$ is a structural zero, with $`P(D_i = 1) = \pi_i`$;
otherwise $`Y_i`$ comes from an $`\text{NB}(\mu_i, \theta)`$
distribution, which can itself produce zeros:
``` math
P(Y_i = 0) = \pi_i + (1-\pi_i)\,f_\text{NB}(0;\mu_i,\theta), \qquad
P(Y_i = y) = (1-\pi_i)\,f_\text{NB}(y;\mu_i,\theta), \quad y \ge 1,
```
where $`f_\text{NB}(0;\mu,\theta) = \{\theta/(\theta+\mu)\}^\theta`$,
$`\log\mu_i = x_i^\top\beta + o_i`$ (count model) and
$`\text{logit}\,\pi_i = z_i^\top\gamma`$ (zero-inflation model). The
covariates $`z_i`$ may differ from $`x_i`$, and the marginal mean is
$`E(Y_i) = (1-\pi_i)\mu_i`$. The observed-data log-likelihood is
``` math
\ell(\beta,\gamma,\theta) = \sum_{i:\,y_i=0}\log\bigl[\pi_i + (1-\pi_i)\,f_\text{NB}(0;\mu_i,\theta)\bigr]
  + \sum_{i:\,y_i>0}\bigl[\log(1-\pi_i) + \log f_\text{NB}(y_i;\mu_i,\theta)\bigr].
```
Because of the mixture, $`\ell`$ is **not** the log-likelihood of an
exponential-family GLM, so IRLS cannot be applied to it directly
(`pscl::zeroinfl()` maximises it with the general-purpose optimiser
BFGS). Instead we use an **EM algorithm** ([Dempster et al.
1977](#ref-dempster1977)) that treats $`D_i`$ as missing data. Given
$`D_i`$, the complete-data log-likelihood splits into a logistic part in
$`\gamma`$ and a weighted NB part in $`(\beta, \theta)`$, and each part
can be fitted by IRLS.

**E-step.** Given the current $`(\beta, \gamma, \theta)`$, compute the
posterior probability that observation $`i`$ is a structural zero:
``` math
\tau_i = P(D_i = 1 \mid y_i) = \begin{cases}
  \dfrac{\pi_i}{\pi_i + (1-\pi_i)\,f_\text{NB}(0;\mu_i,\theta)} & y_i = 0,\\
  0 & y_i > 0.
\end{cases}
```

**M-step (zero part).** Maximise
$`\sum_i[\tau_i\log\pi_i + (1-\tau_i)\log(1-\pi_i)]`$ over $`\gamma`$.
This is a logistic regression with a *fractional* response $`\tau_i`$,
fitted by IRLS with
``` math
w_i = \pi_i(1-\pi_i), \qquad
\tilde z_i = z_i^\top\gamma + \frac{\tau_i - \pi_i}{\pi_i(1-\pi_i)}, \qquad
\gamma \leftarrow (Z^\top W Z)^{-1} Z^\top W \tilde z.
```
(We write $`\tilde z_i`$ for the working response, to distinguish it
from the covariate vector $`z_i`$.)

**M-step (count part).** Maximise
$`\sum_i(1-\tau_i)\log f_\text{NB}(y_i;\mu_i,\theta)`$ over
$`(\beta,\theta)`$. This is exactly the NB fit of Section 2.1 with prior
weights $`a_i = 1-\tau_i`$, so the same IRLS and Newton–Raphson updates
are reused.

**Convergence.** The E- and M-steps are repeated until the observed-data
log-likelihood $`\ell(\beta,\gamma,\theta)`$ changes by less than the
tolerance. EM never decreases $`\ell`$; `zinbreg()` checks this at every
iteration and issues a warning if it ever happens.

**Standard errors.** EM does not produce standard errors. We compute the
observed information matrix as the negative Hessian of our own
observed-data log-likelihood with respect to
$`(\beta, \gamma, \log\theta)`$ at the MLE, using `numDeriv::hessian()`
(numerical differentiation, not optimisation), and invert it. As
`zeroinfl()` does, we report $`\log\hat\theta`$ with its standard error,
and also $`\hat\theta`$.

## 2.3 Algorithms and Convergence Criteria

**NB (`nbreg()`).** $`\beta`$ starts from one Poisson-type weighted
least-squares step: $`\log\max(y_i, 0.5) - o_i`$ is regressed on $`X`$
with weights $`\max(y_i, 0.5)`$, which is a Poisson IRLS step started
from $`\mu_i = y_i`$. $`\theta`$ starts from the moment estimator
$`\hat\theta_0 = \bar y^2 / (s^2 - \bar y)`$ (set to 1 if
$`s^2 \le \bar y`$, and at least 0.1). The outer loop alternates (a)
IRLS for $`\beta`$, iterated until $`\max_j|\Delta\beta_j| < 10^{-10}`$,
and (b) Newton–Raphson for $`\log\theta`$. It stops when the
log-likelihood changes by less than `epsilon` (default $`10^{-8}`$), or
after `maxit` (default 200) outer iterations.

**ZINB (`zinbreg()`).** $`(\beta, \theta)`$ start from an `nbreg()` fit
of the count model. $`\gamma`$ starts with its intercept at the logit of
the observed proportion of zeros and its other coefficients at zero.
Each EM iteration runs one E-step and both M-steps, and the algorithm
stops when $`\ell`$ changes by less than `epsilon` (default
$`10^{-8}`$), or after `maxit` (default 300) iterations. The benchmarks
below use the tighter `epsilon = 1e-10`.

# 3 The R Package

## 3.1 Design and Main Functions

The **nbzinbAmoah** package exports four user-facing functions:

| Function | Purpose |
|----|----|
| `nbreg(formula, data, offset, control, weights)` | NB regression via IRLS + Newton–Raphson for $`\theta`$ |
| `zinbreg(formula, data, offset, control, weights)` | ZINB via EM with IRLS M-steps |
| `nbreg_control(maxit, theta_maxit, epsilon, trace)` | Tolerances and iteration limits for `nbreg` |
| `zinbreg_control(maxit, irls_maxit, theta_maxit, epsilon, trace)` | Tolerances and iteration limits for `zinbreg` |

Both fitting functions return S3 objects (classes `"nbreg"` and
`"zinbreg"`) containing the coefficients, variance–covariance matrix,
$`\hat\theta`$ and its standard error, log-likelihood, number of
iterations, a convergence flag, fitted values and the matched call.
Methods are provided for `print`, `summary` (a coefficient table of
estimates, SEs, $`z`$ values and $`p`$-values, printed by a
`print.summary` method), `coef`, `vcov`, `logLik` (with a `df`
attribute, so `AIC()` and `BIC()` work), `fitted`, `residuals`
(`type = "response"` or `"pearson"`), `nobs`, and `predict` (with
`newdata`). For `zinbreg`, `coef()` and `vcov()` take `model = "count"`,
`"zero"` or `"full"`, and `predict()` supports `type = "response"`
($`(1-\pi)\mu`$), `"count"` ($`\mu`$) and `"zero"` ($`\pi`$).

The `zinbreg` formula uses the two-part syntax `y ~ x1 + x2 | z1 + z2`
from the **Formula** package: count-model covariates before the bar and
zero-model covariates after it. Without `|`, the same covariates are
used in both parts, as in `zeroinfl()`.

## 3.2 Installation

``` r
devtools::install_github("AbigailAmoah/nbzinbAmoah")
library(nbzinbAmoah)
```

## 3.3 Usage Example

``` r
# Fit NB and ZINB on bioChemists (for illustration)
fit_nb_ex <- nbreg(art ~ fem + mar + kid5 + phd + ment,
                   data = bioChemists)
fit_zinb_ex <- zinbreg(art ~ fem + mar + kid5 + phd + ment |
                             fem + mar + kid5 + phd + ment,
                        data = bioChemists)
# Quick look
coef(fit_nb_ex)
```

    (Intercept)    femWomen  marMarried        kid5         phd        ment 
     0.25614402 -0.21641842  0.15048945 -0.17641524  0.01527116  0.02908234 

# 4 Benchmarking

We benchmark **nbzinbAmoah** against `MASS::glm.nb()` (NB) and
`pscl::zeroinfl()` (ZINB) on two datasets: `pscl::bioChemists` (n = 915)
and `AER::NMES1988` (n = 4406). All functions are run with tight
tolerances (`epsilon = 1e-10` for ours and `glm.nb`, `reltol = 1e-12`
for `zeroinfl`). For each fit we compare estimates and standard errors
side by side. For NB the dispersion row is $`\theta`$, and its standard
error for our fit uses the delta method,
$`\text{SE}(\hat\theta) = \hat\theta\,\text{SE}(\log\hat\theta)`$. For
ZINB the dispersion row is $`\log\theta`$, as reported by `zeroinfl`.

## 4.1 bioChemists: NB

|        Term |      Ours | Reference | \|diff\| |  SE ours |   SE ref | \|diff\| SE |
|------------:|----------:|----------:|---------:|---------:|---------:|------------:|
| (Intercept) |  0.256144 |  0.256144 | 3.72e-10 | 0.137348 | 0.137348 |    4.75e-12 |
|    femWomen | -0.216418 | -0.216418 | 6.23e-11 | 0.072636 | 0.072636 |    1.55e-12 |
|  marMarried |  0.150489 |  0.150489 | 1.19e-11 | 0.082097 | 0.082097 |    2.09e-12 |
|        kid5 | -0.176415 | -0.176415 | 4.15e-11 | 0.052813 | 0.052813 |    9.04e-13 |
|         phd |  0.015271 |  0.015271 | 3.75e-11 | 0.035873 | 0.035873 |    8.66e-13 |
|        ment |  0.029082 |  0.029082 | 2.11e-11 | 0.003214 | 0.003214 |    1.82e-13 |
|       theta |  2.264388 |  2.264388 | 8.95e-11 | 0.271176 | 0.271175 |    1.14e-06 |

NB: nbreg vs glm.nb (bioChemists) {.caption-top}

Our nbreg() gives exactly the same results as the built-in glm.nb()
function. The differences are far smaller than the required tolerance.

## 4.2 bioChemists: ZINB

|  Part |        Term |      Ours | Reference | \|diff\| |  SE ours |   SE ref | \|diff\| SE |
|------:|------------:|----------:|----------:|---------:|---------:|---------:|------------:|
| count | (Intercept) |  0.416747 |  0.416747 | 2.75e-07 | 0.143596 | 0.143597 |    5.19e-07 |
| count |    femWomen | -0.195507 | -0.195507 | 2.39e-07 | 0.075593 | 0.075593 |    1.40e-08 |
| count |  marMarried |  0.097583 |  0.097583 | 1.20e-07 | 0.084452 | 0.084452 |    7.79e-09 |
| count |        kid5 | -0.151732 | -0.151732 | 1.74e-07 | 0.054206 | 0.054206 |    7.40e-09 |
| count |         phd | -0.000700 | -0.000700 | 5.28e-08 | 0.036270 | 0.036270 |    1.54e-07 |
| count |        ment |  0.024786 |  0.024786 | 3.10e-08 | 0.003492 | 0.003493 |    2.46e-07 |
| count |  Log(theta) |  0.976358 |  0.976357 | 1.25e-06 | 0.135468 | 0.135469 |    1.45e-06 |
|  zero | (Intercept) | -0.191702 | -0.191688 | 1.35e-05 | 1.322811 | 1.322819 |    7.88e-06 |
|  zero |    femWomen |  0.635934 |  0.635933 | 8.06e-07 | 0.848911 | 0.848918 |    6.74e-06 |
|  zero |  marMarried | -1.499453 | -1.499468 | 1.51e-05 | 0.938662 | 0.938671 |    8.76e-06 |
|  zero |        kid5 |  0.628425 |  0.628427 | 1.97e-06 | 0.442780 | 0.442783 |    2.52e-06 |
|  zero |         phd | -0.037712 | -0.037715 | 2.82e-06 | 0.308006 | 0.308008 |    1.80e-06 |
|  zero |        ment | -0.882286 | -0.882293 | 7.65e-06 | 0.316224 | 0.316228 |    4.08e-06 |

ZINB: zinbreg vs zeroinfl (bioChemists) {.caption-top}

Our zinbreg() also matches zeroinfl() very closely. Both functions find
the same answer, confirming our EM algorithm works correctly.

## 4.3 NMES1988: NB

|            Term |      Ours | Reference | \|diff\| |  SE ours |   SE ref | \|diff\| SE |
|----------------:|----------:|----------:|---------:|---------:|---------:|------------:|
|     (Intercept) |  0.929257 |  0.929257 | 5.97e-11 | 0.054591 | 0.054591 |    3.93e-12 |
|        hospital |  0.217772 |  0.217772 | 7.37e-11 | 0.020176 | 0.020176 |    5.73e-13 |
|      healthpoor |  0.305013 |  0.305013 | 3.96e-11 | 0.048511 | 0.048511 |    2.41e-12 |
| healthexcellent | -0.341807 | -0.341807 | 2.12e-11 | 0.060924 | 0.060924 |    4.28e-12 |
|         chronic |  0.174916 |  0.174916 | 1.79e-11 | 0.012092 | 0.012092 |    7.43e-13 |
|      gendermale | -0.126488 | -0.126488 | 9.37e-12 | 0.031216 | 0.031216 |    2.19e-12 |
|          school |  0.026815 |  0.026815 | 7.52e-13 | 0.004394 | 0.004394 |    3.05e-13 |
|    insuranceyes |  0.224402 |  0.224402 | 1.88e-12 | 0.039464 | 0.039464 |    2.66e-12 |
|           theta |  1.206604 |  1.206604 | 1.36e-10 | 0.033555 | 0.033555 |    4.20e-07 |

NB: nbreg vs glm.nb (NMES1988) {.caption-top}

Our nbreg() matches glm.nb() on the larger NMES1988 dataset too. The
maximum coefficient difference is 7.37e-11 and the log-likelihood
difference is just 1.82e-12 which are similar results.

## 4.4 NMES1988: ZINB

| Part | Term | Ours | Reference | \|diff\| | SE ours | SE ref | \|diff\| SE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| count | (Intercept) | 1.193713 | 1.193713 | 4.92e-07 | 0.056661 | 0.056661 | 1.66e-07 |
| count | hospital | 0.201477 | 0.201477 | 2.66e-08 | 0.020360 | 0.020360 | 8.90e-09 |
| count | healthpoor | 0.285133 | 0.285133 | 8.92e-08 | 0.045093 | 0.045093 | 1.69e-09 |
| count | healthexcellent | -0.319336 | -0.319336 | 2.05e-07 | 0.060405 | 0.060405 | 7.93e-09 |
| count | chronic | 0.129000 | 0.129000 | 1.45e-07 | 0.011931 | 0.011931 | 6.32e-09 |
| count | gendermale | -0.080277 | -0.080277 | 1.34e-07 | 0.031024 | 0.031024 | 7.41e-09 |
| count | school | 0.021423 | 0.021423 | 2.45e-08 | 0.004358 | 0.004358 | 2.00e-08 |
| count | insuranceyes | 0.125864 | 0.125864 | 2.09e-07 | 0.041588 | 0.041588 | 1.15e-08 |
| count | Log(theta) | 0.394144 | 0.394143 | 9.09e-07 | 0.035035 | 0.035035 | 9.51e-08 |
| zero | (Intercept) | -0.046884 | -0.046879 | 5.44e-06 | 0.268561 | 0.268558 | 2.93e-06 |
| zero | hospital | -0.800488 | -0.800483 | 4.23e-06 | 0.420817 | 0.420816 | 1.30e-06 |
| zero | chronic | -1.247934 | -1.247938 | 4.76e-06 | 0.178316 | 0.178316 | 4.11e-07 |
| zero | gendermale | 0.647678 | 0.647678 | 3.97e-07 | 0.200114 | 0.200115 | 7.08e-07 |
| zero | school | -0.083770 | -0.083771 | 6.35e-07 | 0.026254 | 0.026253 | 7.17e-07 |
| zero | insuranceyes | -1.175611 | -1.175615 | 4.17e-06 | 0.220119 | 0.220121 | 1.98e-06 |

ZINB: zinbreg vs zeroinfl (NMES1988) {.caption-top}

Our zinbreg() also agrees closely with zeroinfl() on NMES1988. The
maximum coefficient difference is 4.92e-7, whivh is within our tolerance
of $`10^{-3}`$

## 4.5 Agreement Summary

[Table 1](#tbl-agree) gives the maximum absolute difference for each
quantity in Table 4 of the project description, and whether all four are
within the required tolerance ($`10^{-5}`$ for NB coefficients and SEs,
$`10^{-4}`$ for $`\theta`$ and the NB log-likelihood, $`10^{-3}`$ for
everything in ZINB).

| Dataset | Model | Coef | SE | theta / log(theta) | logLik | Within tolerance |
|:---|:---|:---|:---|:---|:---|:---|
| bioChemists | NB | 3.72e-10 | 4.75e-12 | 8.95e-11 | 2.27e-13 | yes |
| bioChemists | ZINB | 1.51e-05 | 8.76e-06 | 1.25e-06 | 4.37e-10 | yes |
| NMES1988 | NB | 7.37e-11 | 4.28e-12 | 1.36e-10 | 1.82e-12 | yes |
| NMES1988 | ZINB | 5.44e-06 | 2.93e-06 | 9.09e-07 | 9.97e-10 | yes |

Table 1: Maximum absolute differences between nbzinbAmoah and the
reference implementations

All differences are within the required tolerances. This confirms that
our package gives the same results as trusted reference functions across
all four datasets and models.

## 4.6 Computation Time

Each fit is timed with `system.time()`, taking the median elapsed time
of 5 repetitions. Our code is not expected to be faster than the
reference.

| Dataset     | Model | Ours (s) | Reference (s) | Ratio |
|:------------|:------|---------:|--------------:|------:|
| bioChemists | NB    |     0.02 |          0.10 |   0.2 |
| bioChemists | ZINB  |     1.47 |          0.32 |   4.6 |
| NMES1988    | NB    |     0.10 |          0.34 |   0.3 |
| NMES1988    | ZINB  |     5.73 |          1.44 |   4.0 |

Median elapsed time over 5 repetitions {.caption-top}

Our NB function is actually faster than the built-in one (0.02s vs 0.09s
on bioChemists). Our ZINB function is slower, it takes more steps to
reach the answer. This is expected because the EM algorithm works
differently from the optimizer used by zeroinfl().

# 5 Simulation Study

We assess the finite-sample performance of `zinbreg()` by simulating
$`R = 100`$ datasets of size $`n = 500`$ from a ZINB model with known
parameters.

**True parameters:** $`\beta = (0.5, 0.3, -0.4)`$,
$`\gamma = (-1.0, 0.8)`$, $`\theta = 2.0`$.

| Parameter  |   Truth |    Bias |  MC_SD | Mean_SE | Coverage_95 |
|:-----------|--------:|--------:|-------:|--------:|------------:|
| beta_0     |  0.5000 | -0.0004 | 0.0912 |  0.0878 |        0.93 |
| beta_1     |  0.3000 |  0.0044 | 0.0596 |  0.0613 |        0.95 |
| beta_2     | -0.4000 | -0.0085 | 0.0667 |  0.0618 |        0.93 |
| gamma_0    | -1.0000 | -0.0217 | 0.3582 |  0.3449 |        0.95 |
| gamma_1    |  0.8000 |  0.0621 | 0.2620 |  0.2428 |        0.93 |
| log(theta) |  0.6931 |  0.0598 | 0.3196 |  0.3094 |        0.95 |

Simulation results (R=100 replicates, n=500). Converged: 99/100.
{.caption-top}

The largest bias across all parameters is 0.062, for gamma_1, which is
small relative to the true value of 0.8. The 95% confidence intervals
captured the true value between 93% and 95% of the time across all
parameters, close to the expected 95%. This tells us that zinbreg()
gives reliable estimates and accurate standard errors.

# 6 Data Analysis: NMES1988

We analyse the `NMES1988` dataset from the **AER** package ([Kleiber and
Zeileis 2008](#ref-kleiber2008)), which records physician office visits
for 4,406 individuals aged 66 and over from the US National Medical
Expenditure Survey (1987–88) ([Deb and Trivedi 1997](#ref-deb1997)).

## 6.1 Descriptive Summary

|    n |   mean | variance | prop_zero | poisson_pred_zero |
|-----:|-------:|---------:|----------:|------------------:|
| 4406 | 5.7744 |  45.6871 |     0.155 |            0.0031 |

Descriptive statistics for physician office visits (NMES1988)
{.caption-top}

![](data:image/png;base64,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)

Observed visit counts (bars) vs Poisson expected (points). The Poisson
severely under-predicts zeros and over-predicts moderate counts.

The variance (45.69) far exceeds the mean (5.77), indicating
overdispersion. The observed proportion of zeros (0.155) exceeds the
Poisson prediction (0.003), suggesting excess zeros.

## 6.2 Model Fitting

We fit a Poisson GLM with `glm()` (for reference only), and NB and ZINB
models with **nbzinbAmoah**. As specified in the project description,
the count model uses `hospital`, `health`, `chronic`, `gender`, `school`
and `insurance`, and the zero model uses `hospital`, `chronic`,
`gender`, `school` and `insurance`.

## 6.3 Model Comparison

| Model   |  df |   logLik |     AIC |     BIC | ExpZeros |
|:--------|----:|---------:|--------:|--------:|---------:|
| Poisson |   8 | -17971.6 | 35959.2 | 36010.4 |     46.7 |
| NB      |   9 | -12170.6 | 24359.1 | 24416.6 |    608.0 |
| ZINB    |  15 | -12090.7 | 24211.4 | 24307.3 |    708.9 |

Model comparison for NMES1988 (observed zeros = 683) {.caption-top}

## 6.4 Interpretation

| Covariate       |   IRR | 95% CI         |
|:----------------|------:|:---------------|
| hospital        | 1.243 | (1.195, 1.293) |
| healthpoor      | 1.357 | (1.234, 1.492) |
| healthexcellent | 0.710 | (0.631, 0.801) |
| chronic         | 1.191 | (1.163, 1.220) |
| gendermale      | 0.881 | (0.829, 0.937) |
| school          | 1.027 | (1.018, 1.036) |
| insuranceyes    | 1.252 | (1.158, 1.352) |

NB model: incidence rate ratios (IRR) with 95% Wald CIs. Values above 1
indicate higher visit rates. {.caption-top}

| Covariate    |    OR | 95% CI         | p-value |
|:-------------|------:|:---------------|:--------|
| hospital     | 0.449 | (0.197, 1.025) | 0.0571  |
| chronic      | 0.287 | (0.202, 0.407) | \<1e-04 |
| gendermale   | 1.911 | (1.291, 2.829) | 0.0012  |
| school       | 0.920 | (0.874, 0.968) | 0.0014  |
| insuranceyes | 0.309 | (0.200, 0.475) | \<1e-04 |

ZINB zero part: odds ratios for being a structural zero {.caption-top}

**Dispersion.** The NB model estimates $`\hat\theta`$ = 1.207 (SE of
$`\log\hat\theta`$ = 0.028). At the mean visit count of 5.77, the NB
variance $`\mu + \mu^2/\theta`$ is therefore about 33.4, compared with
5.77 under the Poisson model. The ZINB count part has $`\log\hat\theta`$
= 0.394 (SE 0.035), $`\hat\theta`$ = 1.483.

For the NB model each additional hospital stay is associated with 24%
more physician visits. People in poor health make 36% more visits than
people in average health. Those in excellent health make 29% fewer
visits. Each chronic condition is associated with 19% more visits. Men
make 12% fewer visits than women. Having insurance is associated with
25% more visits.

The ZINB count part shows the same pattern as the NB model; hospital
stays, poor health, chronic conditions, and insurance all increase
visits, while excellent health and being male reduce visits. The effects
are slightly smaller because some zeros are now handled by the zero
part.

The zero part shows who never visits a doctor. People with chronic
conditions are less likely to be non-visitors (OR = 0.287) because they
have no choice but to seek care. Men are almost twice as likely as women
to never visit a doctor (OR = 1.911). People with insurance are much
less likely to be non-visitors (OR = 0.309), suggesting cost keeps some
people away. More education also slightly reduces the chance of never
visiting (OR = 0.920). Hospital stays were not significant with p =
0.057.

## 6.5 Model Choice

The Poisson model fits very poorly. Its log-likelihood is 5801 lower
than the NB model’s, and it predicts only 47 zeros against 683 observed.
Allowing overdispersion fixes most of the problem, but the NB model
still under-predicts zeros (608 expected). The ZINB model improves the
log-likelihood by a further 79.8 with 6 extra parameters, predicts 709
zeros, and has the lowest AIC **and** the lowest BIC.

We therefore prefer the **ZINB model** for NMES1988. For the bioChemists
data in Appendix A of the project description, BIC favoured the simpler
NB model; here, the evidence for excess zeros is strong enough to
survive BIC’s larger penalty. The zero part also adds a useful
interpretation: it identifies a group of structural non-users of
physician care, who are more likely to be male, uninsured, less educated
and to have fewer chronic conditions.

# 7 Discussion

This project implemented NB and ZINB regression from scratch, using only
IRLS and Newton–Raphson (NB) and an EM algorithm with IRLS M-steps
(ZINB). No general-purpose optimizer was used in the fitting code.

The benchmarking results confirm that our implementation reproduces
`MASS::glm.nb()` to within 3.72e-10 and `pscl::zeroinfl()` to within
1.51e-05 for coefficients, with log-likelihood differences of at most
9.97e-10. These are well within the required tolerances ($`10^{-5}`$ for
NB and $`10^{-3}`$ for ZINB coefficients).

The simulation study ($`R`$ = 100, $`n`$ = 500) shows that `zinbreg()`
reliably recovers known ZINB parameters with negligible bias and
well-calibrated 95% Wald intervals (empirical coverage 0.93–0.95).

**Limitations.** First, the NB standard errors for $`\beta`$ treat
$`\theta`$ as known, as `glm.nb()` does, so they ignore the uncertainty
in $`\hat\theta`$. Second, the ZINB standard errors use a numerical
Hessian, which is slower and can be less accurate than an analytic one.
Third, EM converges linearly, so `zinbreg()` needs many iterations and
is several times slower than `zeroinfl()` (see the timing table).
Fourth, offsets are supported in the count model only, not in the zero
model. Finally, in the NMES1988 analysis the zero-model covariates
follow the project description, and the choice between NB and ZINB
relies on information criteria rather than a formal test, because the
usual likelihood-ratio test is non-standard when the null value
$`\pi_i = 0`$ lies on the boundary of the parameter space.

**Possible extensions** include an analytic Hessian for the ZINB
standard errors, an offset in the zero model, acceleration of the EM
algorithm (for example, switching to Newton–Raphson near convergence),
and a formal score test for zero inflation.

# 8 AI-Use Statement

I used Claude.ai together with claude code and vscode to write the main
functions, nbreg() and zinbreg(), including the IRLS, Newton-Raphson,
and EM algorithms. It also helped me set up the GitHub repository,
prepare the benchmark scripts, run the simulation study, create the
pkgdown website, and draft the report. I followed the process by
providing instructions based on the equations and requirements in the
project description.

I read the project description to understand the statistical methods. I
also checked the benchmark comparisons results agreed with
MASS::glm.nb() and pscl::zeroinfl() within the required tolerances. I
reviewed the report, checked the results, and made changes to the
wording where necessary.

I noticed a function that Claude used inside the package that caused an
error during testing. I read the error message, and asked Claude to use
a different approach, “solve(crossprod())”. That actually worked.

# 9 References

Deb, Partha, and Pravin K. Trivedi. 1997. “Demand for Medical Care by
the Elderly: A Finite Mixture Approach.” *Journal of Applied
Econometrics* 12 (3): 313–36.

Dempster, A. P., N. M. Laird, and D. B. Rubin. 1977. “Maximum Likelihood
from Incomplete Data via the EM Algorithm.” *Journal of the Royal
Statistical Society, Series B* 39 (1): 1–38.

Kleiber, Christian, and Achim Zeileis. 2008. *Applied Econometrics with
R*. Springer.

Lambert, Diane. 1992. “Zero-Inflated Poisson Regression, with an
Application to Defects in Manufacturing.” *Technometrics* 34 (1): 1–14.

McCullagh, P., and J. A. Nelder. 1989. *Generalized Linear Models*. 2nd
ed. Chapman & Hall.

Venables, W. N., and B. D. Ripley. 2002. *Modern Applied Statistics with
S*. 4th ed. Springer.

Zeileis, Achim, Christian Kleiber, and Simon Jackman. 2008. “Regression
Models for Count Data in R.” *Journal of Statistical Software* 27 (8):
1–25.

# 10 Appendix A: Full Model Output for bioChemists

The `pscl::bioChemists` dataset records the number of articles published
by 915 PhD biochemists in their last three years.

## 10.1 NB Model

    Negative Binomial Regression
    Call: nbreg(formula = f_bio_nb, data = bioChemists, control = ctrl_nb)

    Coefficients:
                 Estimate Std. Error z value Pr(>|z|)    
    (Intercept)  0.256144   0.137348   1.865 0.062191 .  
    femWomen    -0.216418   0.072636  -2.979 0.002887 ** 
    marMarried   0.150489   0.082097   1.833 0.066791 .  
    kid5        -0.176415   0.052813  -3.340 0.000837 ***
    phd          0.015271   0.035873   0.426 0.670326    
    ment         0.029082   0.003214   9.048  < 2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Theta: 2.2644  (SE of log-theta: 0.1198)
    Log-likelihood: -1560.9583 on 908 Df
    n = 915  Converged: TRUE  Iterations: 4

## 10.2 ZINB Model

    Zero-Inflated Negative Binomial Regression
    Call: zinbreg(formula = f_bio_zinb, data = bioChemists, control = ctrl_zinb)


    Count model coefficients (NB with log link):
                  Estimate Std. Error z value Pr(>|z|)    
    (Intercept)  0.4167468  0.1435960   2.902  0.00371 ** 
    femWomen    -0.1955066  0.0755925  -2.586  0.00970 ** 
    marMarried   0.0975827  0.0844519   1.155  0.24789    
    kid5        -0.1517323  0.0542061  -2.799  0.00512 ** 
    phd         -0.0007001  0.0362695  -0.019  0.98460    
    ment         0.0247862  0.0034924   7.097 1.27e-12 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Log(theta): 0.9764  (SE: 0.1355)  =>  theta = 2.6548

    Zero-inflation model coefficients (binomial with logit link):
                Estimate Std. Error z value Pr(>|z|)   
    (Intercept) -0.19170    1.32281  -0.145  0.88477   
    femWomen     0.63593    0.84891   0.749  0.45379   
    marMarried  -1.49945    0.93866  -1.597  0.11017   
    kid5         0.62843    0.44278   1.419  0.15582   
    phd         -0.03771    0.30801  -0.122  0.90255   
    ment        -0.88229    0.31622  -2.790  0.00527 **
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Log-likelihood: -1549.9909 on 902 Df
    n = 915  Converged: TRUE  Iterations: 135
