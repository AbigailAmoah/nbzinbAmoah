## Benchmarking: compare nbzinbAmoah with MASS::glm.nb and pscl::zeroinfl
## Run from package root: Rscript inst/benchmark/benchmark.R

library(nbzinbAmoah)
library(MASS)
library(pscl)
library(AER)

cat("============================================================\n")
cat(" nbzinbAmoah Benchmarking Report\n")
cat(" Datasets: pscl::bioChemists  and  AER::NMES1988\n")
cat("============================================================\n\n")

## ---- Helper: agreement table ----
agreement_table <- function(ours_coef, ref_coef, ours_se, ref_se,
                             ours_lth, ref_lth, ours_ll, ref_ll,
                             label_ref = "glm.nb") {
  cat(sprintf("  Max |coef diff|     : %.2e  (tol 1e-5)\n",
              max(abs(ours_coef - ref_coef))))
  cat(sprintf("  Max |SE diff|       : %.2e  (tol 1e-5)\n",
              max(abs(ours_se - ref_se))))
  cat(sprintf("  |log(theta) diff|   : %.2e  (tol 1e-4)\n",
              abs(ours_lth - ref_lth)))
  cat(sprintf("  |loglik diff|       : %.2e  (tol 1e-4)\n\n",
              abs(ours_ll - ref_ll)))
}

## ============================================================
## 1. bioChemists — NB
## ============================================================
cat("--- 1. bioChemists: NB (nbreg vs glm.nb) ---\n")
data("bioChemists", package = "pscl")

t_ours <- system.time(
  fit_nb_ours <- nbreg(art ~ fem + mar + kid5 + phd + ment,
                       data = bioChemists,
                       control = nbreg_control(maxit = 500, epsilon = 1e-10))
)
t_ref <- system.time(
  fit_nb_ref <- MASS::glm.nb(art ~ fem + mar + kid5 + phd + ment,
                              data = bioChemists,
                              control = glm.control(epsilon = 1e-10, maxit = 100))
)

agreement_table(
  coef(fit_nb_ours), coef(fit_nb_ref),
  sqrt(diag(vcov(fit_nb_ours))), sqrt(diag(vcov(fit_nb_ref))),
  log(fit_nb_ours$theta), log(fit_nb_ref$theta),
  as.numeric(logLik(fit_nb_ours)), as.numeric(logLik(fit_nb_ref))
)
cat(sprintf("  Time (ours / ref): %.3fs / %.3fs\n\n",
            t_ours["elapsed"], t_ref["elapsed"]))

## Side-by-side coefficient table
cat("  Coefficient comparison:\n")
tab_nb <- data.frame(
  `nbreg`   = round(coef(fit_nb_ours), 6),
  `glm.nb`  = round(coef(fit_nb_ref),  6),
  `diff`    = round(coef(fit_nb_ours) - coef(fit_nb_ref), 2e-7)
)
print(tab_nb)

## ============================================================
## 2. bioChemists — ZINB
## ============================================================
cat("\n--- 2. bioChemists: ZINB (zinbreg vs zeroinfl) ---\n")

t_ours2 <- system.time(
  fit_zinb_ours <- zinbreg(art ~ fem + mar + kid5 + phd + ment |
                                 fem + mar + kid5 + phd + ment,
                            data = bioChemists,
                            control = zinbreg_control(maxit = 500, epsilon = 1e-10))
)
t_ref2 <- system.time(
  fit_zinb_ref <- pscl::zeroinfl(art ~ fem + mar + kid5 + phd + ment |
                                       fem + mar + kid5 + phd + ment,
                                  data = bioChemists, dist = "negbin",
                                  control = zeroinfl.control(reltol = 1e-12))
)

ref_cnt <- coef(fit_zinb_ref, model = "count")
ref_cnt <- ref_cnt[names(ref_cnt) != "Log(theta)"]
ref_zer <- coef(fit_zinb_ref, model = "zero")

cat("  Count part:\n")
tab_zinb_c <- data.frame(
  `zinbreg` = round(coef(fit_zinb_ours, "count"), 6),
  `zeroinfl`= round(ref_cnt, 6),
  `diff`    = round(coef(fit_zinb_ours, "count") - ref_cnt, 2e-7)
)
print(tab_zinb_c)

cat("  Zero part:\n")
tab_zinb_z <- data.frame(
  `zinbreg` = round(coef(fit_zinb_ours, "zero"), 6),
  `zeroinfl`= round(ref_zer, 6),
  `diff`    = round(coef(fit_zinb_ours, "zero") - ref_zer, 2e-7)
)
print(tab_zinb_z)

cat(sprintf("\n  log(theta): ours=%.6f  ref=%.6f  diff=%.2e\n",
            log(fit_zinb_ours$theta),
            as.numeric(ref_cnt["Log(theta)"] %||% coef(fit_zinb_ref)["count.(Log(theta))"]),
            abs(log(fit_zinb_ours$theta) - coef(fit_zinb_ref, "count")["Log(theta)"])))
cat(sprintf("  loglik:     ours=%.4f  ref=%.4f  diff=%.2e\n",
            as.numeric(logLik(fit_zinb_ours)),
            as.numeric(logLik(fit_zinb_ref)),
            abs(as.numeric(logLik(fit_zinb_ours)) - as.numeric(logLik(fit_zinb_ref)))))
cat(sprintf("  Time (ours / ref): %.3fs / %.3fs\n\n",
            t_ours2["elapsed"], t_ref2["elapsed"]))

## ============================================================
## 3. NMES1988 — NB
## ============================================================
cat("--- 3. NMES1988: NB (nbreg vs glm.nb) ---\n")
data("NMES1988", package = "AER")

t3 <- system.time(
  fit_nmes_nb <- nbreg(
    visits ~ hospital + health + chronic + gender + school + insurance,
    data = NMES1988, control = nbreg_control(maxit = 500, epsilon = 1e-10))
)
t3r <- system.time(
  fit_nmes_ref <- MASS::glm.nb(
    visits ~ hospital + health + chronic + gender + school + insurance,
    data = NMES1988, control = glm.control(epsilon = 1e-10, maxit = 100))
)
cat(sprintf("  Max |coef diff|: %.2e   |theta diff|: %.2e   |ll diff|: %.2e\n",
            max(abs(coef(fit_nmes_nb) - coef(fit_nmes_ref))),
            abs(fit_nmes_nb$theta - fit_nmes_ref$theta),
            abs(as.numeric(logLik(fit_nmes_nb)) - as.numeric(logLik(fit_nmes_ref)))))
cat(sprintf("  Time (ours / ref): %.3fs / %.3fs\n\n", t3["elapsed"], t3r["elapsed"]))

## ============================================================
## 4. NMES1988 — ZINB
## ============================================================
cat("--- 4. NMES1988: ZINB (zinbreg vs zeroinfl) ---\n")
t4 <- system.time(
  fit_nmes_zinb <- zinbreg(
    visits ~ hospital + health + chronic + gender + school + insurance |
             hospital + chronic + gender + school + insurance,
    data = NMES1988, control = zinbreg_control(maxit = 500, epsilon = 1e-10))
)
t4r <- system.time(
  fit_nmes_zinb_ref <- pscl::zeroinfl(
    visits ~ hospital + health + chronic + gender + school + insurance |
             hospital + chronic + gender + school + insurance,
    data = NMES1988, dist = "negbin",
    control = zeroinfl.control(reltol = 1e-12))
)
rcc <- coef(fit_nmes_zinb_ref, "count"); rcc <- rcc[names(rcc) != "Log(theta)"]
cat(sprintf("  Max |count coef diff|: %.2e   |ll diff|: %.2e\n",
            max(abs(coef(fit_nmes_zinb, "count") - rcc)),
            abs(as.numeric(logLik(fit_nmes_zinb)) - as.numeric(logLik(fit_nmes_zinb_ref)))))
cat(sprintf("  Time (ours / ref): %.3fs / %.3fs\n\n", t4["elapsed"], t4r["elapsed"]))

cat("============================================================\n")
cat(" Done.\n")

## Null-coalescing operator (simple local version)
`%||%` <- function(a, b) if (!is.null(a) && !is.na(a)) a else b
