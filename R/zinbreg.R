#' Control parameters for zinbreg
#'
#' @param maxit Maximum EM iterations (default 300).
#' @param irls_maxit Max IRLS iterations within each M-step (default 50).
#' @param theta_maxit Max Newton-Raphson iterations for theta (default 50).
#' @param epsilon Convergence tolerance on observed log-likelihood (default 1e-8).
#' @param trace Logical; if TRUE print log-likelihood at each EM iteration.
#' @return A list of control parameters.
#' @export
zinbreg_control <- function(maxit = 300, irls_maxit = 50, theta_maxit = 50,
                             epsilon = 1e-8, trace = FALSE) {
  list(maxit = maxit, irls_maxit = irls_maxit,
       theta_maxit = theta_maxit, epsilon = epsilon, trace = trace)
}

#' Zero-Inflated Negative Binomial Regression
#'
#' Fits a zero-inflated negative binomial (ZINB) regression via an EM algorithm
#' whose M-steps are iteratively reweighted least squares (IRLS).
#' No general-purpose optimizer is used.
#'
#' The model has two components:
#' \itemize{
#'   \item \strong{Count part}: \eqn{\log \mu_i = x_i^\top \beta + o_i}
#'   \item \strong{Zero part}: \eqn{\text{logit}\,\pi_i = z_i^\top \gamma}
#' }
#' Specify both parts with a two-part formula \code{y ~ x1 + x2 | z1 + z2}.
#' If no \code{|} is present, the same covariates are used for both parts.
#'
#' @param formula A two-part formula \code{y ~ count_covariates | zero_covariates}.
#'   If \code{|} is omitted, the same covariates are used in both parts.
#' @param data A data frame.
#' @param offset An optional offset for the count part (log scale).
#' @param control A list from \code{\link{zinbreg_control}}.
#' @param weights Optional prior weights.
#' @return An object of class \code{"zinbreg"} with components:
#'   \item{coefficients}{List with \code{count} (beta) and \code{zero} (gamma).}
#'   \item{theta}{Estimated dispersion parameter.}
#'   \item{loglik}{Observed-data log-likelihood at convergence.}
#'   \item{vcov}{Variance-covariance matrix (numerical Hessian of observed loglik).}
#'   \item{fitted.values}{Expected counts \eqn{(1-\pi_i)\mu_i}.}
#'   \item{fitted.pi}{Estimated zero-inflation probabilities \eqn{\pi_i}.}
#'   \item{fitted.mu}{Estimated NB means \eqn{\mu_i}.}
#'   \item{residuals}{Response residuals \eqn{y - (1-\pi)\mu}.}
#'   \item{nobs}{Number of observations.}
#'   \item{df.residual}{Residual degrees of freedom.}
#'   \item{iter}{Number of EM iterations.}
#'   \item{converged}{Logical.}
#'   \item{call}{The matched call.}
#' @examples
#' if (requireNamespace("pscl", quietly = TRUE)) {
#'   data("bioChemists", package = "pscl")
#'   fit <- zinbreg(art ~ fem + mar + kid5 + phd + ment |
#'                        fem + mar + kid5 + phd + ment,
#'                  data = bioChemists)
#'   summary(fit)
#' }
#' @importFrom Formula Formula
#' @importFrom stats model.frame model.matrix model.response model.offset
#'   model.weights plogis pnorm dnbinom
#' @export
zinbreg <- function(formula, data, offset = NULL,
                    control = zinbreg_control(), weights = NULL) {
  cl <- match.call()

  ## ---- 1. Parse two-part formula via Formula package ----
  f   <- Formula::Formula(formula)
  ## Count part: left-hand side ~ right-hand side part 1
  ## Zero part:  left-hand side ~ right-hand side part 2 (or part 1 if none)
  mf  <- model.frame(f, data = data)
  y   <- model.response(mf, "numeric")
  n   <- length(y)

  X   <- model.matrix(f, data = mf, rhs = 1)   # count covariates
  Z   <- if (length(f)[2] >= 2)
           model.matrix(f, data = mf, rhs = 2) # zero covariates
         else X

  off <- model.offset(mf)
  if (is.null(off)) off <- rep(0, n)
  if (!is.null(offset)) off <- off + offset

  a <- model.weights(mf)
  if (is.null(a)) a <- rep(1, n)
  if (!is.null(weights)) a <- a * weights

  p <- ncol(X); q <- ncol(Z)
  is_zero <- (y == 0)

  ## ---- 2. Starting values: NB fit for (beta, theta), logistic for gamma ----
  ## Fit NB to get starting beta and theta
  nb_start <- tryCatch(
    nbreg(as.formula(paste("y ~", paste(colnames(X)[-1], collapse = "+"))),
          data = as.data.frame(cbind(y = y, X[, -1, drop = FALSE])),
          control = nbreg_control(maxit = 50, trace = FALSE)),
    error = function(e) NULL
  )
  if (!is.null(nb_start) && nb_start$converged) {
    beta  <- nb_start$coefficients
    theta <- nb_start$theta
  } else {
    beta  <- rep(0, p); beta[1] <- log(mean(y) + 0.1)
    theta <- 1.0
  }
  names(beta) <- colnames(X)

  ## Starting gamma: intercept-only logistic from observed zeros
  pi0   <- mean(is_zero)
  gamma <- rep(0, q)
  gamma[1] <- log(pi0 / (1 - pi0 + 1e-6))

  ## ---- 3. EM loop ----
  loglik_obs <- function(beta, gamma, theta) {
    mu  <- exp(as.numeric(X %*% beta) + off)
    pi  <- plogis(as.numeric(Z %*% gamma))
    ## For y = 0: log(pi + (1-pi)*f_NB(0; mu, theta))
    ## For y > 0: log(1-pi) + log f_NB(y; mu, theta)
    f0  <- dnbinom(0, size = theta, mu = mu)
    ll  <- numeric(n)
    ll[is_zero]  <- log(pi[is_zero]  + (1 - pi[is_zero])  * f0[is_zero] + 1e-300)
    ll[!is_zero] <- log(1 - pi[!is_zero]) +
                    dnbinom(y[!is_zero], size = theta, mu = mu[!is_zero], log = TRUE)
    sum(a * ll)
  }

  loglik_old <- loglik_obs(beta, gamma, theta)
  converged  <- FALSE

  for (iter in seq_len(control$maxit)) {

    mu <- exp(as.numeric(X %*% beta) + off)
    pi <- plogis(as.numeric(Z %*% gamma))

    ## -- E-step: posterior probability of being structural zero --
    ## tau_i = P(D_i=1 | y_i) =
    ##   pi_i / (pi_i + (1-pi_i)*f_NB(0; mu_i, theta))  if y_i = 0
    ##   0                                                 if y_i > 0
    ## (Section 3.2 of project description)
    f0  <- dnbinom(0, size = theta, mu = mu)
    tau <- numeric(n)
    denom <- pi * is_zero + (1 - pi) * f0   # denominator for zeros
    tau[is_zero]  <- pi[is_zero] / (pi[is_zero] + (1 - pi[is_zero]) * f0[is_zero] + 1e-300)
    tau[!is_zero] <- 0

    ## -- M-step (zero part): logistic regression by IRLS --
    ## Maximize sum_i [ tau_i * log(pi_i) + (1-tau_i)*log(1-pi_i) ]
    ## with pi_i = plogis(z_i'gamma)
    ## This is weighted logistic regression with fractional response tau_i
    ## IRLS weights:  w_i = pi_i * (1 - pi_i)                (a_i-weighted: use a*w)
    ## Working resp:  z_i = z_i'gamma + (tau_i - pi_i) / (pi_i*(1-pi_i))
    for (irls_z in seq_len(control$irls_maxit)) {
      eta_z <- as.numeric(Z %*% gamma)
      pi_c  <- plogis(eta_z)
      w_z   <- a * pi_c * (1 - pi_c)
      z_z   <- eta_z + (tau - pi_c) / (pi_c * (1 - pi_c) + 1e-14)
      ## Weighted logistic IRLS: gamma = (Z'WZ)^{-1} Z'Wz
      ZtWZ <- crossprod(Z, (w_z + 1e-14) * Z)
      ZtWz <- crossprod(Z, (w_z + 1e-14) * z_z)
      gamma_new <- tryCatch(
        as.numeric(solve(ZtWZ, ZtWz)),
        error = function(e) gamma
      )
      if (any(!is.finite(gamma_new))) break
      if (max(abs(gamma_new - gamma)) < 1e-10) { gamma <- gamma_new; break }
      gamma <- gamma_new
    }

    ## -- M-step (count part): weighted NB fit --
    ## Maximize sum_i (1-tau_i) * log f_NB(y_i; mu_i, theta)
    ## = NB fit with prior weights a_i * (1 - tau_i)
    ## (Section 3.2: "reuse your NB code" with a_i = 1 - tau_i)
    a_nb <- a * (1 - tau)   # effective weights for the NB part

    ## IRLS for beta (NB, same as nbreg inner loop) with theta fixed
    for (irls_b in seq_len(control$irls_maxit)) {
      eta_b <- as.numeric(X %*% beta) + off
      mu    <- exp(eta_b)
      w_b   <- a_nb * theta * mu / (theta + mu)
      z_b   <- (eta_b - off) + (y - mu) / mu
      W_b   <- sqrt(w_b + 1e-14)
      Xw    <- X * W_b; zw_b <- z_b * W_b
      XtWX_b <- crossprod(X, (w_b + 1e-14) * X)
      XtWz_b <- crossprod(X, (w_b + 1e-14) * z_b)
      beta_new <- tryCatch(as.numeric(solve(XtWX_b, XtWz_b)),
                           error = function(e) beta)
      if (any(!is.finite(beta_new))) break
      if (max(abs(beta_new - beta)) < 1e-10) { beta <- beta_new; break }
      beta <- beta_new
    }

    ## NR for theta with current beta and weights a_nb
    phi <- log(theta)
    mu  <- exp(as.numeric(X %*% beta) + off)
    for (nr_i in seq_len(control$theta_maxit)) {
      theta <- exp(phi)
      U <- sum(a_nb * (digamma(y + theta) - digamma(theta) +
                         log(theta) + 1 - log(theta + mu) -
                         (y + theta)/(theta + mu)))
      H <- sum(a_nb * (trigamma(y + theta) - trigamma(theta) +
                         1/theta - 2/(theta + mu) +
                         (y + theta)/(theta + mu)^2))
      U_phi <- theta * U
      H_phi <- theta^2 * H + theta * U
      if (abs(H_phi) < 1e-14) break
      phi_new <- phi - U_phi / H_phi
      phi_new <- max(phi_new, -10)
      if (abs(phi_new - phi) < 1e-10) { phi <- phi_new; break }
      phi <- phi_new
    }
    theta <- max(exp(phi), 1e-6)

    ## -- Convergence check on observed-data log-likelihood --
    loglik_new <- loglik_obs(beta, gamma, theta)

    if (control$trace)
      message(sprintf("EM iter %d: loglik = %.6f", iter, loglik_new))

    ## EM must never decrease the log-likelihood
    if (loglik_new < loglik_old - 1e-6)
      warning(sprintf("EM log-likelihood decreased at iteration %d (%.8f -> %.8f)",
                      iter, loglik_old, loglik_new))

    if (abs(loglik_new - loglik_old) < control$epsilon) {
      converged <- TRUE
      loglik_old <- loglik_new
      break
    }
    loglik_old <- loglik_new
  }

  ## ---- 4. Standard errors via numerical Hessian of observed log-likelihood ----
  ## Parameter vector: (beta, gamma, log_theta)
  ## We use numDeriv::hessian if available, else a simple finite-difference
  par_vec <- c(beta, gamma, log(theta))
  npar <- length(par_vec)
  p_b <- length(beta); q_g <- length(gamma)

  ll_par <- function(par) {
    b  <- par[seq_len(p_b)]
    g  <- par[p_b + seq_len(q_g)]
    th <- exp(par[p_b + q_g + 1])
    if (th <= 0 || !is.finite(th)) return(-1e15)
    loglik_obs(b, g, th)
  }

  H_obs <- tryCatch({
    if (requireNamespace("numDeriv", quietly = TRUE)) {
      numDeriv::hessian(ll_par, par_vec)
    } else {
      ## Finite-difference fallback (less accurate but no dependency)
      fd_hessian(ll_par, par_vec)
    }
  }, error = function(e) matrix(NA_real_, npar, npar))

  vcov_all <- tryCatch(
    solve(-H_obs),
    error = function(e) matrix(NA_real_, npar, npar)
  )

  ## Split vcov back into components
  idx_b    <- seq_len(p_b)
  idx_g    <- p_b + seq_len(q_g)
  idx_lth  <- p_b + q_g + 1
  vcov_beta  <- vcov_all[idx_b,   idx_b,   drop = FALSE]
  vcov_gamma <- vcov_all[idx_g,   idx_g,   drop = FALSE]
  SE_logtheta <- if (!is.na(vcov_all[idx_lth, idx_lth]))
                   sqrt(max(vcov_all[idx_lth, idx_lth], 0)) else NA_real_

  rownames(vcov_beta)  <- colnames(vcov_beta)  <- colnames(X)
  rownames(vcov_gamma) <- colnames(vcov_gamma) <- colnames(Z)

  ## Final fitted values
  mu_f <- exp(as.numeric(X %*% beta) + off)
  pi_f <- plogis(as.numeric(Z %*% gamma))

  structure(
    list(
      coefficients  = list(count = beta, zero = gamma),
      theta         = theta,
      SE_logtheta   = SE_logtheta,
      loglik        = loglik_old,
      vcov          = list(count = vcov_beta, zero = vcov_gamma, full = vcov_all),
      fitted.values = (1 - pi_f) * mu_f,
      fitted.pi     = pi_f,
      fitted.mu     = mu_f,
      residuals     = y - (1 - pi_f) * mu_f,
      nobs          = n,
      df.residual   = n - p_b - q_g - 1L,
      iter          = iter,
      converged     = converged,
      call          = cl,
      terms         = list(count = terms(formula(f, rhs = 1)),
                           zero  = terms(formula(f, rhs = if (length(f)[2] >= 2) 2 else 1))),
      model         = mf,
      X = X, Z = Z, off = off, a = a
    ),
    class = "zinbreg"
  )
}

## ---- Finite-difference Hessian (fallback if numDeriv not available) ----
fd_hessian <- function(f, x, h = 1e-4) {
  n <- length(x)
  H <- matrix(0, n, n)
  f0 <- f(x)
  for (i in seq_len(n)) {
    for (j in i:n) {
      ei <- ej <- numeric(n)
      ei[i] <- h; ej[j] <- h
      H[i, j] <- (f(x + ei + ej) - f(x + ei) - f(x + ej) + f0) / h^2
      H[j, i] <- H[i, j]
    }
  }
  H
}

## ---- S3 methods for zinbreg ----

#' @export
print.zinbreg <- function(x, digits = 4, ...) {
  cat("\nZero-Inflated Negative Binomial Regression (EM with IRLS M-steps)\n")
  cat("Call: "); print(x$call); cat("\n")
  cat("Count model coefficients:\n")
  print(round(x$coefficients$count, digits))
  cat("Zero model coefficients:\n")
  print(round(x$coefficients$zero, digits))
  cat(sprintf("\nTheta: %.4f\nLog-likelihood: %.4f\nIterations: %d  Converged: %s\n",
              x$theta, x$loglik, x$iter, x$converged))
  invisible(x)
}

#' @export
summary.zinbreg <- function(object, ...) {
  ## Count part
  se_c  <- sqrt(pmax(diag(object$vcov$count), 0))
  zc    <- object$coefficients$count / se_c
  pc    <- 2 * pnorm(abs(zc), lower.tail = FALSE)
  coef_count <- cbind(Estimate = object$coefficients$count,
                      `Std. Error` = se_c,
                      `z value` = zc,
                      `Pr(>|z|)` = pc)
  ## Zero part
  se_z  <- sqrt(pmax(diag(object$vcov$zero), 0))
  zz    <- object$coefficients$zero / se_z
  pz    <- 2 * pnorm(abs(zz), lower.tail = FALSE)
  coef_zero <- cbind(Estimate = object$coefficients$zero,
                     `Std. Error` = se_z,
                     `z value` = zz,
                     `Pr(>|z|)` = pz)
  structure(
    list(call = object$call, coef_count = coef_count, coef_zero = coef_zero,
         theta = object$theta, SE_logtheta = object$SE_logtheta,
         loglik = object$loglik, nobs = object$nobs,
         df.residual = object$df.residual,
         converged = object$converged, iter = object$iter),
    class = "summary.zinbreg"
  )
}

#' @export
print.summary.zinbreg <- function(x, digits = 4, signif.stars = TRUE, ...) {
  cat("\nZero-Inflated Negative Binomial Regression\n")
  cat("Call: "); print(x$call); cat("\n")
  cat("\nCount model coefficients (NB with log link):\n")
  stats::printCoefmat(x$coef_count, digits = digits, signif.stars = signif.stars)
  cat(sprintf("\nLog(theta): %.4f  (SE: %.4f)  =>  theta = %.4f\n",
              log(x$theta), x$SE_logtheta, x$theta))
  cat("\nZero-inflation model coefficients (binomial with logit link):\n")
  stats::printCoefmat(x$coef_zero, digits = digits, signif.stars = signif.stars)
  cat(sprintf("\nLog-likelihood: %.4f on %d Df\n", x$loglik, x$df.residual))
  cat(sprintf("n = %d  Converged: %s  Iterations: %d\n",
              x$nobs, x$converged, x$iter))
  invisible(x)
}

#' @export
coef.zinbreg <- function(object, model = c("full", "count", "zero"), ...) {
  model <- match.arg(model)
  if (model == "full") {
    c(object$coefficients$count, object$coefficients$zero)
  } else {
    object$coefficients[[model]]
  }
}

#' @export
vcov.zinbreg <- function(object, model = c("count", "zero", "full"), ...) {
  model <- match.arg(model)
  object$vcov[[model]]
}

#' @export
logLik.zinbreg <- function(object, ...) {
  df <- length(object$coefficients$count) +
        length(object$coefficients$zero) + 1L
  structure(object$loglik, df = df, class = "logLik")
}

#' @export
fitted.zinbreg <- function(object, ...) object$fitted.values

#' @export
residuals.zinbreg <- function(object, type = c("response", "pearson"), ...) {
  type <- match.arg(type)
  if (type == "response") return(object$residuals)
  mu  <- object$fitted.values
  object$residuals / sqrt(mu)
}

#' @export
#' @method nobs zinbreg
nobs.zinbreg <- function(object, ...) object$nobs

#' @export
predict.zinbreg <- function(object, newdata = NULL,
                             type = c("response", "count", "zero"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    mu <- object$fitted.mu
    pi <- object$fitted.pi
  } else {
    X2  <- model.matrix(object$terms$count, data = newdata)
    Z2  <- model.matrix(object$terms$zero,  data = newdata)
    off2 <- rep(0, nrow(newdata))
    mu  <- exp(as.numeric(X2 %*% object$coefficients$count) + off2)
    pi  <- plogis(as.numeric(Z2 %*% object$coefficients$zero))
  }
  switch(type,
    response = (1 - pi) * mu,
    count    = mu,
    zero     = pi
  )
}
