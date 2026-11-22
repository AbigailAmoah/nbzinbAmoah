# Suppress R CMD check notes about global variable bindings
utils::globalVariables(c("var", "as.formula", "terms"))

## ---- Warnings required by Section 4.3 of the project description ----

## As theta -> Inf the NB variance mu + mu^2/theta tends to the Poisson
## variance mu. Above this value the extra term mu^2/theta is less than 1%
## of mu whenever mu < 100, so the data show essentially no overdispersion.
theta_poisson_limit <- 1e4

#' Warn when theta is so large that the NB model is effectively Poisson
#' @param theta Estimated dispersion parameter.
#' @noRd
warn_large_theta <- function(theta) {
  if (is.finite(theta) && theta > theta_poisson_limit)
    warning(sprintf(paste0(
      "theta estimate is very large (%.3g): the data show little or no ",
      "overdispersion (theta -> Inf is the Poisson limit), so a Poisson ",
      "model may be adequate"), theta), call. = FALSE)
  invisible(NULL)
}

#' Warn when the outer iteration stopped at maxit without converging
#' @param converged Logical convergence flag.
#' @param maxit Maximum number of iterations allowed.
#' @param what Name of the algorithm, used in the message.
#' @noRd
warn_not_converged <- function(converged, maxit, what) {
  if (!converged)
    warning(sprintf(paste0(
      "%s did not converge within maxit = %d iterations; results may be ",
      "unreliable. Increase maxit in the control argument."), what, maxit),
      call. = FALSE)
  invisible(NULL)
}
