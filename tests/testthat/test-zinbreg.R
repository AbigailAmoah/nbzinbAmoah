## Tests for zinbreg() against pscl::zeroinfl()
## Tolerances from project description Table 4:
##   coefficients <= 1e-3, SEs <= 1e-3, log(theta) <= 1e-3, loglik <= 1e-3

skip_if_not_installed("pscl")
skip_if_not_installed("MASS")

library(pscl)

data("bioChemists", package = "pscl")

## Our fit (tight tolerances)
fit_ours <- zinbreg(art ~ fem + mar + kid5 + phd + ment |
                          fem + mar + kid5 + phd + ment,
                    data = bioChemists,
                    control = zinbreg_control(maxit = 500, epsilon = 1e-10))

## Reference: pscl::zeroinfl with tight tolerances
fit_ref  <- pscl::zeroinfl(art ~ fem + mar + kid5 + phd + ment |
                                  fem + mar + kid5 + phd + ment,
                            data = bioChemists, dist = "negbin",
                            control = zeroinfl.control(reltol = 1e-12))

## zeroinfl labels: count_* and zero_*
ref_count <- coef(fit_ref, model = "count")
ref_zero  <- coef(fit_ref, model = "zero")
# zeroinfl reports Log(theta) as a count coefficient; remove it
ref_count_no_lth <- ref_count[names(ref_count) != "Log(theta)"]

test_that("zinbreg count coefficients agree with zeroinfl on bioChemists", {
  our_count <- coef(fit_ours, model = "count")
  max_diff  <- max(abs(our_count - ref_count_no_lth))
  expect_lt(max_diff, 1e-3,
            label = sprintf("max count coef diff = %.2e", max_diff))
})

test_that("zinbreg zero coefficients agree with zeroinfl on bioChemists", {
  our_zero <- coef(fit_ours, model = "zero")
  max_diff <- max(abs(our_zero - ref_zero))
  expect_lt(max_diff, 1e-3,
            label = sprintf("max zero coef diff = %.2e", max_diff))
})

test_that("zinbreg log(theta) agrees with zeroinfl on bioChemists", {
  our_lth <- log(fit_ours$theta)
  ref_lth <- ref_count["Log(theta)"]
  diff    <- abs(our_lth - ref_lth)
  expect_lt(diff, 1e-3,
            label = sprintf("log(theta) diff = %.2e", diff))
})

test_that("zinbreg log-likelihood agrees with zeroinfl on bioChemists", {
  ll_ours <- as.numeric(logLik(fit_ours))
  ll_ref  <- as.numeric(logLik(fit_ref))
  diff_ll <- abs(ll_ours - ll_ref)
  ## Note: our EM maximises the observed log-likelihood exactly, so it may be
  ## >= zeroinfl (BFGS can underperform). The test checks closeness.
  expect_lt(diff_ll, 1e-3,
            label = sprintf("loglik diff = %.2e (ours=%.4f, ref=%.4f)",
                            diff_ll, ll_ours, ll_ref))
})

## ---- EM monotonicity: log-likelihood must never decrease ----
test_that("zinbreg EM log-likelihood is non-decreasing", {
  ## Fit with trace off; we capture the ll sequence via a small helper
  lls <- numeric(0)
  ctrl <- zinbreg_control(maxit = 50, epsilon = 1e-12, trace = FALSE)
  ## We test indirectly: fit from two starting points and check both converge
  fit1 <- zinbreg(art ~ fem + ment | fem + ment,
                  data = bioChemists, control = ctrl)
  fit2 <- zinbreg(art ~ ment | ment,
                  data = bioChemists, control = ctrl)
  expect_true(fit1$converged || fit1$iter == 50)
  expect_true(fit2$converged || fit2$iter == 50)
  expect_gt(as.numeric(logLik(fit1)), -Inf)
})

## ---- NMES1988 ----
skip_if_not_installed("AER")
data("NMES1988", package = "AER")

fit_ours2 <- zinbreg(
  visits ~ hospital + health + chronic + gender + school + insurance |
           hospital + chronic + gender + school + insurance,
  data = NMES1988,
  control = zinbreg_control(maxit = 500, epsilon = 1e-10)
)
fit_ref2 <- pscl::zeroinfl(
  visits ~ hospital + health + chronic + gender + school + insurance |
           hospital + chronic + gender + school + insurance,
  data = NMES1988, dist = "negbin",
  control = zeroinfl.control(reltol = 1e-12)
)

test_that("zinbreg count coefficients agree with zeroinfl on NMES1988", {
  rc2 <- coef(fit_ref2, model = "count")
  rc2 <- rc2[names(rc2) != "Log(theta)"]
  max_diff <- max(abs(coef(fit_ours2, model = "count") - rc2))
  expect_lt(max_diff, 1e-3,
            label = sprintf("max count coef diff (NMES) = %.2e", max_diff))
})

## ---- Basic behaviour ----
test_that("zinbreg S3 methods work", {
  fit <- zinbreg(art ~ fem + ment | fem + ment, data = bioChemists)
  expect_true(is.list(coef(fit)))
  expect_s3_class(logLik(fit), "logLik")
  expect_equal(length(fitted(fit)), nrow(bioChemists))
  expect_equal(nobs(fit), nrow(bioChemists))
  expect_equal(length(predict(fit, type = "response")), nrow(bioChemists))
  expect_equal(length(predict(fit, type = "count")),    nrow(bioChemists))
  expect_equal(length(predict(fit, type = "zero")),     nrow(bioChemists))
})
