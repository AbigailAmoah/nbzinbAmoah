## Tests for nbreg() against MASS::glm.nb()
## Tolerances from project description Table 4:
##   coefficients <= 1e-5, SEs <= 1e-5, theta <= 1e-4, loglik <= 1e-4

skip_if_not_installed("MASS")
skip_if_not_installed("pscl")

library(MASS)
library(pscl)

data("bioChemists", package = "pscl")

## Run our implementation with tight tolerances
fit_ours <- nbreg(art ~ fem + mar + kid5 + phd + ment,
                  data = bioChemists,
                  control = nbreg_control(maxit = 500, epsilon = 1e-10))

## Reference: MASS::glm.nb with tight tolerances
fit_ref  <- MASS::glm.nb(art ~ fem + mar + kid5 + phd + ment,
                          data = bioChemists,
                          control = glm.control(epsilon = 1e-10, maxit = 100))

test_that("nbreg coefficients agree with MASS::glm.nb on bioChemists", {
  max_diff_coef <- max(abs(coef(fit_ours) - coef(fit_ref)))
  expect_lt(max_diff_coef, 1e-5,
            label = sprintf("max coef diff = %.2e", max_diff_coef))
})

test_that("nbreg standard errors agree with MASS::glm.nb on bioChemists", {
  se_ours <- sqrt(diag(vcov(fit_ours)))
  se_ref  <- sqrt(diag(vcov(fit_ref)))
  max_diff_se <- max(abs(se_ours - se_ref))
  expect_lt(max_diff_se, 1e-5,
            label = sprintf("max SE diff = %.2e", max_diff_se))
})

test_that("nbreg theta agrees with MASS::glm.nb on bioChemists", {
  diff_theta <- abs(fit_ours$theta - fit_ref$theta)
  expect_lt(diff_theta, 1e-4,
            label = sprintf("theta diff = %.2e", diff_theta))
})

test_that("nbreg log-likelihood agrees with MASS::glm.nb on bioChemists", {
  ll_ours <- as.numeric(logLik(fit_ours))
  ll_ref  <- as.numeric(logLik(fit_ref))
  diff_ll <- abs(ll_ours - ll_ref)
  expect_lt(diff_ll, 1e-4,
            label = sprintf("loglik diff = %.2e", diff_ll))
})

## ---- NMES1988 dataset ----
skip_if_not_installed("AER")
data("NMES1988", package = "AER")

fit_ours2 <- nbreg(visits ~ hospital + health + chronic + gender + school + insurance,
                   data = NMES1988,
                   control = nbreg_control(maxit = 500, epsilon = 1e-10))
fit_ref2  <- MASS::glm.nb(visits ~ hospital + health + chronic + gender + school + insurance,
                           data = NMES1988,
                           control = glm.control(epsilon = 1e-10, maxit = 100))

test_that("nbreg coefficients agree with MASS::glm.nb on NMES1988", {
  max_diff <- max(abs(coef(fit_ours2) - coef(fit_ref2)))
  expect_lt(max_diff, 1e-5,
            label = sprintf("max coef diff = %.2e", max_diff))
})

test_that("nbreg theta agrees with MASS::glm.nb on NMES1988", {
  diff_th <- abs(fit_ours2$theta - fit_ref2$theta)
  expect_lt(diff_th, 1e-4,
            label = sprintf("theta diff = %.2e", diff_th))
})

## ---- Basic behaviour ----
test_that("nbreg handles offset correctly", {
  set.seed(42)
  n <- 100
  df <- data.frame(y = rpois(n, 2), x = rnorm(n), off = log(runif(n, 0.5, 2)))
  fit <- nbreg(y ~ x, data = df, offset = df$off,
               control = nbreg_control(maxit = 200))
  expect_true(fit$converged)
  expect_equal(length(coef(fit)), 2)
})

test_that("nbreg single-covariate model converges", {
  set.seed(1)
  df <- data.frame(y = rnbinom(200, size = 2, mu = 3), x = rnorm(200))
  fit <- nbreg(y ~ x, data = df)
  expect_true(fit$converged)
  expect_true(fit$theta > 0)
})

test_that("nbreg S3 methods work", {
  fit <- nbreg(art ~ fem + ment, data = bioChemists)
  expect_true(is.numeric(coef(fit)))
  expect_true(is.matrix(vcov(fit)))
  expect_s3_class(logLik(fit), "logLik")
  expect_equal(length(fitted(fit)), nrow(bioChemists))
  expect_equal(length(residuals(fit)), nrow(bioChemists))
  expect_equal(nobs(fit), nrow(bioChemists))
  pred <- predict(fit, type = "response")
  expect_equal(length(pred), nrow(bioChemists))
})

## ---- Section 4.3 warnings ----
test_that("nbreg warns when maxit is reached without converging", {
  expect_warning(
    nbreg(art ~ fem + ment, data = bioChemists,
          control = nbreg_control(maxit = 1)),
    "did not converge")
})

test_that("nbreg warns when theta is very large (Poisson limit)", {
  set.seed(1)
  x <- rnorm(500)
  d <- data.frame(x = x, y = rpois(500, exp(1 + 0.3 * x)))  # no overdispersion
  expect_warning(nbreg(y ~ x, data = d), "Poisson")
})
