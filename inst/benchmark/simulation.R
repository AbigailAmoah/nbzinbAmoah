## Simulation study for zinbreg()
## R = 100 replicates, n = 500, two covariates in each part
## Run from package root: Rscript inst/benchmark/simulation.R

library(nbzinbAmoah)

set.seed(2026)

R   <- 100     # number of replicates
n   <- 500     # sample size

## True parameters
beta_true  <- c(0.5, 0.3, -0.4)   # (intercept, x1, x2) for count part
gamma_true <- c(-1.0, 0.8)         # (intercept, z1) for zero part
theta_true <- 2.0

cat("============================================================\n")
cat(" Simulation Study: zinbreg() parameter recovery\n")
cat(sprintf(" R = %d, n = %d\n", R, n))
cat(sprintf(" True beta  = (%s)\n", paste(beta_true,  collapse = ", ")))
cat(sprintf(" True gamma = (%s)\n", paste(gamma_true, collapse = ", ")))
cat(sprintf(" True theta = %.2f\n\n", theta_true))

## Storage
results <- data.frame(
  b0 = NA_real_, b1 = NA_real_, b2 = NA_real_,
  g0 = NA_real_, g1 = NA_real_,
  lth = NA_real_,
  se_b0 = NA_real_, se_b1 = NA_real_, se_b2 = NA_real_,
  se_g0 = NA_real_, se_g1 = NA_real_,
  se_lth = NA_real_,
  converged = NA
)[rep(1, R), ]

ctrl <- zinbreg_control(maxit = 400, epsilon = 1e-9, trace = FALSE)

for (r in seq_len(R)) {
  ## Simulate covariates
  x1 <- rnorm(n); x2 <- rnorm(n)
  z1 <- rnorm(n)

  ## True mean and zero probability
  X  <- cbind(1, x1, x2)
  Z  <- cbind(1, z1)
  mu  <- exp(X %*% beta_true)
  pi  <- plogis(Z %*% gamma_true)

  ## Generate ZINB data
  D  <- rbinom(n, 1, pi)         # structural zero indicator
  y  <- ifelse(D == 1, 0, rnbinom(n, size = theta_true, mu = mu))

  df <- data.frame(y = y, x1 = x1, x2 = x2, z1 = z1)

  fit <- tryCatch(
    zinbreg(y ~ x1 + x2 | z1, data = df, control = ctrl),
    error = function(e) NULL
  )
  if (is.null(fit)) next

  results[r, "b0"]  <- coef(fit, "count")[1]
  results[r, "b1"]  <- coef(fit, "count")[2]
  results[r, "b2"]  <- coef(fit, "count")[3]
  results[r, "g0"]  <- coef(fit, "zero")[1]
  results[r, "g1"]  <- coef(fit, "zero")[2]
  results[r, "lth"] <- log(fit$theta)

  se <- tryCatch(sqrt(pmax(diag(vcov(fit, "full")), 0)), error = function(e) rep(NA, 6))
  results[r, "se_b0"]  <- se[1]
  results[r, "se_b1"]  <- se[2]
  results[r, "se_b2"]  <- se[3]
  results[r, "se_g0"]  <- se[4]
  results[r, "se_g1"]  <- se[5]
  results[r, "se_lth"] <- se[6]

  results[r, "converged"] <- fit$converged
}

## Drop failed replicates
ok <- !is.na(results$b0)
cat(sprintf("Converged/successful: %d / %d\n\n", sum(ok, na.rm = TRUE), R))
res <- results[ok, ]

## True values on log scale for theta
true_lth <- log(theta_true)
truth <- c(beta_true, gamma_true, true_lth)
names(truth) <- c("b0","b1","b2","g0","g1","lth")

## Compute bias, MC SD, mean estimated SE, empirical 95% coverage
param_names <- c("b0","b1","b2","g0","g1","lth")
se_names    <- c("se_b0","se_b1","se_b2","se_g0","se_g1","se_lth")

out <- do.call(rbind, lapply(seq_along(param_names), function(i) {
  pn <- param_names[i]; sn <- se_names[i]
  est  <- res[[pn]];    se  <- res[[sn]]
  tr   <- truth[pn]
  bias <- mean(est - tr, na.rm = TRUE)
  mc_sd <- sd(est, na.rm = TRUE)
  mean_se <- mean(se, na.rm = TRUE)
  ## Wald 95% CI coverage
  cov <- mean(abs(est - tr) / se < qnorm(0.975), na.rm = TRUE)
  data.frame(Parameter = pn, Truth = tr, Bias = bias,
             MC_SD = mc_sd, Mean_SE = mean_se, Coverage_95 = cov)
}))

cat("Simulation Results:\n")
print(out, digits = 4, row.names = FALSE)
cat("\n(Coverage_95 should be near 0.95)\n")

## Save results
saveRDS(list(out = out, results = results),
        file = "inst/benchmark/simulation_results.rds")
cat("\nResults saved to inst/benchmark/simulation_results.rds\n")
