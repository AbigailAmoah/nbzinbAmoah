#' Control parameters for nbreg
#'
#' @param maxit Maximum number of outer IRLS iterations (default 200).
#' @param theta_maxit Maximum Newton-Raphson iterations for theta (default 50).
#' @param epsilon Convergence tolerance on log-likelihood change (default 1e-8).
#' @param trace Logical; if TRUE print log-likelihood at each iteration.
#' @return A list of control parameters.
#' @export
nbreg_control <- function(maxit = 200, theta_maxit = 50,
                           epsilon = 1e-8, trace = FALSE) {
  list(maxit = maxit, theta_maxit = theta_maxit,
       epsilon = epsilon, trace = trace)
}

#' Negative Binomial Regression
#'
#' Fits a negative binomial (NB2) regression model using iteratively reweighted
#' least squares (IRLS) for the coefficient vector beta, alternated with
#' Newton-Raphson updates for the dispersion parameter theta.
#' No general-purpose optimizer is used.
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}.
#' @param data A data frame containing the variables in the formula.
#' @param offset An optional numeric vector of offsets on the log-mean scale.
#'   Can also be specified in the formula via \code{offset()}.
#' @param control A list of control parameters from \code{\link{nbreg_control}}.
#' @param weights Optional prior weights (all 1 by default).
#' @return An object of class \code{"nbreg"} with components:
#'   \item{coefficients}{Named vector of estimated beta coefficients.}
#'   \item{theta}{Estimated dispersion parameter theta > 0.}
#'   \item{SE_theta}{Standard error of log(theta).}
#'   \item{loglik}{Maximised log-likelihood.}
#'   \item{vcov}{Variance-covariance matrix of beta (treating theta as known).}
#'   \item{fitted.values}{Fitted means mu_i.}
#'   \item{residuals}{Response residuals y - mu.}
#'   \item{weights}{Prior weights used.}
#'   \item{nobs}{Number of observations.}
#'   \item{df.residual}{Residual degrees of freedom.}
#'   \item{iter}{Number of outer iterations.}
#'   \item{converged}{Logical; TRUE if converged within maxit.}
#'   \item{call}{The matched call.}
#'   \item{terms}{The terms object.}
#'   \item{model}{The model frame.}
#' @examples
#' if (requireNamespace("pscl", quietly = TRUE)) {
#'   data("bioChemists", package = "pscl")
#'   fit <- nbreg(art ~ fem + mar + kid5 + phd + ment, data = bioChemists)
#'   summary(fit)
#' }
#' @importFrom stats model.frame model.matrix model.response model.offset
#'   model.weights qr qr.coef pnorm
#' @export
nbreg <- function(formula, data, offset = NULL, control = nbreg_control(),
                  weights = NULL) {
  cl <- match.call()

  ## ---- 1. Build model frame, design matrix, response ----
  mf <- model.frame(formula, data = data)
  y  <- model.response(mf, "numeric")
  X  <- model.matrix(formula, data = mf)
  n  <- length(y)
  p  <- ncol(X)

  ## Offset
  off <- model.offset(mf)
  if (is.null(off)) off <- rep(0, n)
  if (!is.null(offset)) off <- off + offset

  ## Prior weights (a_i in the project description)
  a <- model.weights(mf)
  if (is.null(a)) a <- rep(1, n)
  if (!is.null(weights)) a <- a * weights

  ## ---- 2. Starting values ----
  ## Start beta from a Poisson IRLS (one outer pass using large theta)
  ## Use log(y + 0.1) as initial eta
  eta  <- log(pmax(y, 0.5)) - off          # initial linear predictor
  mu   <- exp(eta + off)                   # initial mu

  ## Initial theta: moment estimator (clipped to be safe)
  v    <- var(y)
  mbar <- mean(y)
  theta <- if (v > mbar) mbar^2 / (v - mbar) else 1.0
  theta <- max(theta, 0.1)

  ## Starting beta from weighted least squares with initial mu
  beta <- tryCatch(
    solve(crossprod(X, X * a * mu), crossprod(X, a * mu * (log(pmax(y, 0.5)) - off))),
    error = function(e) rep(0, p)
  )
  beta <- as.numeric(beta)
  names(beta) <- colnames(X)

  loglik_old <- -Inf
  converged  <- FALSE

  ## ---- 3. Outer loop: alternate IRLS (beta) and NR (theta) ----
  for (iter in seq_len(control$maxit)) {

    ## -- 3a. IRLS for beta with theta fixed --
    ## Equation: w_i = a_i * (theta * mu_i) / (theta + mu_i)
    ##           z_i = (eta_i - off_i) + (y_i - mu_i) / mu_i   [working response]
    ##           beta <- (X'WX)^{-1} X'Wz
    for (irls_iter in seq_len(50)) {
      eta <- as.numeric(X %*% beta) + off
      mu  <- exp(eta)

      ## IRLS weights: w_i = a_i * theta * mu_i / (theta + mu_i)
      ## Variance of NB2: V(mu) = mu + mu^2/theta
      ## Fisher weight = a_i * (d mu/d eta)^2 / V(mu) = a_i * mu^2 / (mu + mu^2/theta)
      ##               = a_i * theta * mu / (theta + mu)
      w <- a * theta * mu / (theta + mu)

      ## Working response: z_i = (eta_i - off_i) + (y_i - mu_i) / mu_i
      z <- (eta - off) + (y - mu) / mu

      ## Weighted least squares: beta_new = (X'WX)^{-1} X'Wz
      W    <- sqrt(w)
      Xw   <- X * W
      zw   <- z * W
      qrXw <- base::qr(Xw)
      beta_new <- as.numeric(base::qr.coef(qrXw, zw))

      delta_beta <- max(abs(beta_new - beta))
      beta <- beta_new
      if (delta_beta < 1e-10) break
    }

    ## Final mu after IRLS
    eta <- as.numeric(X %*% beta) + off
    mu  <- exp(eta)

    ## -- 3b. Newton-Raphson for theta (on phi = log theta for stability) --
    ## Score:    U(theta) = sum_i a_i [ psi(y_i + theta) - psi(theta)
    ##                                  + log(theta) + 1 - log(theta + mu_i)
    ##                                  - (y_i + theta)/(theta + mu_i) ]
    ## Hessian:  H(theta) = sum_i a_i [ psi'(y_i+theta) - psi'(theta)
    ##                                  + 1/theta - 2/(theta+mu_i)
    ##                                  + (y_i+theta)/(theta+mu_i)^2 ]
    ## Reparameterise: phi = log theta  =>
    ##   U_phi = theta * U,   H_phi = theta^2 * H + theta * U
    phi <- log(theta)
    for (nr_iter in seq_len(control$theta_maxit)) {
      theta <- exp(phi)
      U <- sum(a * (digamma(y + theta) - digamma(theta) +
                      log(theta) + 1 - log(theta + mu) -
                      (y + theta) / (theta + mu)))
      H <- sum(a * (trigamma(y + theta) - trigamma(theta) +
                      1/theta - 2/(theta + mu) +
                      (y + theta) / (theta + mu)^2))
      ## On log scale: U_phi = theta*U,  H_phi = theta^2*H + theta*U
      U_phi <- theta * U
      H_phi <- theta^2 * H + theta * U
      if (abs(H_phi) < 1e-14) break
      phi_new <- phi - U_phi / H_phi
      phi_new <- max(phi_new, -10)   # keep theta > 0
      delta_phi <- abs(phi_new - phi)
      phi <- phi_new
      if (delta_phi < 1e-10) break
    }
    theta <- exp(phi)

    ## -- 3c. Compute log-likelihood and check convergence --
    ## ell = sum_i a_i [ log Gamma(y_i+theta) - log Gamma(theta) - log(y_i!)
    ##                   + theta*log(theta) + y_i*log(mu_i)
    ##                   - (theta + y_i)*log(theta + mu_i) ]
    loglik <- sum(a * (lgamma(y + theta) - lgamma(theta) - lgamma(y + 1) +
                         theta * log(theta) + y * log(mu) -
                         (theta + y) * log(theta + mu)))

    if (control$trace)
      message(sprintf("Iter %d: loglik = %.6f, theta = %.6f", iter, loglik, theta))

    if (abs(loglik - loglik_old) < control$epsilon) {
      converged <- TRUE
      break
    }
    loglik_old <- loglik
  }

  ## ---- 4. Standard errors for beta ----
  ## Var(beta_hat) = (X' W_hat X)^{-1}  (treating theta as known, as glm.nb does)
  w_final <- a * theta * mu / (theta + mu)
  W_final <- X * sqrt(w_final)
  vcov_beta <- tryCatch(
    chol2inv(chol(crossprod(W_final))),
    error = function(e) solve(crossprod(W_final))
  )
  rownames(vcov_beta) <- colnames(vcov_beta) <- colnames(X)

  ## SE of log(theta): SE = 1 / sqrt(-H_phi)  (observed information)
  theta_final <- theta
  U_check <- sum(a * (digamma(y + theta_final) - digamma(theta_final) +
                        log(theta_final) + 1 - log(theta_final + mu) -
                        (y + theta_final) / (theta_final + mu)))
  H_check <- sum(a * (trigamma(y + theta_final) - trigamma(theta_final) +
                        1/theta_final - 2/(theta_final + mu) +
                        (y + theta_final) / (theta_final + mu)^2))
  H_phi_final <- theta_final^2 * H_check + theta_final * U_check
  SE_theta <- if (-H_phi_final > 0) 1 / sqrt(-H_phi_final) else NA_real_

  ## ---- 5. Assemble output ----
  structure(
    list(
      coefficients  = beta,
      theta         = theta_final,
      SE_theta      = SE_theta,
      loglik        = loglik,
      vcov          = vcov_beta,
      fitted.values = mu,
      residuals     = y - mu,
      weights       = a,
      nobs          = n,
      df.residual   = n - p - 1L,
      iter          = iter,
      converged     = converged,
      call          = cl,
      terms         = attr(mf, "terms"),
      model         = mf
    ),
    class = "nbreg"
  )
}

## ---- NB log-likelihood (internal, used by zinbreg) ----
nb_loglik <- function(y, mu, theta, a = rep(1, length(y))) {
  sum(a * (lgamma(y + theta) - lgamma(theta) - lgamma(y + 1) +
             theta * log(theta) + y * log(mu) -
             (theta + y) * log(theta + mu)))
}

## ---- S3 methods for nbreg ----

#' @export
print.nbreg <- function(x, digits = 4, ...) {
  cat("\nNegative Binomial Regression (IRLS + Newton-Raphson)\n")
  cat("Call: "); print(x$call); cat("\n")
  cat("Coefficients:\n")
  print(round(x$coefficients, digits))
  cat(sprintf("\nTheta: %.4f\nLog-likelihood: %.4f\nIterations: %d  Converged: %s\n",
              x$theta, x$loglik, x$iter, x$converged))
  invisible(x)
}

#' @export
summary.nbreg <- function(object, ...) {
  se   <- sqrt(diag(object$vcov))
  zval <- object$coefficients / se
  pval <- 2 * pnorm(abs(zval), lower.tail = FALSE)
  coeftab <- cbind(
    Estimate   = object$coefficients,
    `Std. Error` = se,
    `z value`  = zval,
    `Pr(>|z|)` = pval
  )
  structure(
    list(call = object$call, coeftab = coeftab,
         theta = object$theta, SE_theta = object$SE_theta,
         loglik = object$loglik, nobs = object$nobs,
         df.residual = object$df.residual,
         converged = object$converged, iter = object$iter),
    class = "summary.nbreg"
  )
}

#' @export
print.summary.nbreg <- function(x, digits = 4, signif.stars = TRUE, ...) {
  cat("\nNegative Binomial Regression\n")
  cat("Call: "); print(x$call); cat("\n")
  cat("Coefficients:\n")
  stats::printCoefmat(x$coeftab, digits = digits, signif.stars = signif.stars)
  cat(sprintf("\nTheta: %.4f  (SE of log-theta: %.4f)\n", x$theta, x$SE_theta))
  cat(sprintf("Log-likelihood: %.4f on %d Df\n", x$loglik, x$df.residual))
  cat(sprintf("n = %d  Converged: %s  Iterations: %d\n",
              x$nobs, x$converged, x$iter))
  invisible(x)
}

#' @export
coef.nbreg <- function(object, ...) object$coefficients

#' @export
vcov.nbreg <- function(object, ...) object$vcov

#' @export
logLik.nbreg <- function(object, ...) {
  p <- length(object$coefficients) + 1L  # +1 for theta
  structure(object$loglik, df = p, class = "logLik")
}

#' @export
fitted.nbreg <- function(object, ...) object$fitted.values

#' @export
residuals.nbreg <- function(object, type = c("response", "pearson"), ...) {
  type <- match.arg(type)
  if (type == "response") return(object$residuals)
  ## Pearson: (y - mu) / sqrt(V(mu))  where V(mu) = mu + mu^2/theta
  mu  <- object$fitted.values
  th  <- object$theta
  object$residuals / sqrt(mu + mu^2 / th)
}

#' @export
nobs.nbreg <- function(object, ...) object$nobs

#' @export
predict.nbreg <- function(object, newdata = NULL, type = c("response", "link"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    eta <- log(object$fitted.values)
    mu  <- object$fitted.values
  } else {
    X   <- model.matrix(object$terms, data = newdata)
    off <- model.offset(model.frame(object$terms, data = newdata))
    if (is.null(off)) off <- rep(0, nrow(newdata))
    eta <- as.numeric(X %*% object$coefficients) + off
    mu  <- exp(eta)
  }
  if (type == "link") eta else mu
}
