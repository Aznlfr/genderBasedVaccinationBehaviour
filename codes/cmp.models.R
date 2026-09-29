library(tidyr)
library(dplyr)
library(here)

nsamples <- 47 * 2
nstates <- 51

# Convert logical values to the T/F labels used by ensemble.py.
tf_label <- function(x) ifelse(x, "T", "F")

# Number of fitted mechanistic parameters per state in ensemble.py:
#   alpha, cv, cvbar, tilde_c_i, k1                         -> 5 baseline
#   separate_alpha, separate_cv, separate_cvbar, fit_sigma -> +1 each when TRUE
#
# The deviance below profiles one residual variance separately for each state,
# so add one more nuisance parameter per state for the AIC/AICc calculation.
AIC_dete <- function(mdl, Dev, fit_sigma, separate_cv,
                     separate_cvbar, separate_alpha) {
  n_model_par_state <- 5 +
    as.integer(fit_sigma) +
    as.integer(separate_cv) +
    as.integer(separate_cvbar) +
    as.integer(separate_alpha)

  npar_state <- n_model_par_state + 1
  npar_total <- npar_state * nstates
  nobs_total <- nsamples * nstates

  AIC <- Dev + 2 * npar_total
  AICc <- AIC + 2 * npar_total * (npar_total + 1) /
    (nobs_total - npar_total - 1)
  BIC <- Dev + npar_total*log(nobs_total)
  
  tibble(
    model = mdl,
    fit_sigma = fit_sigma,
    separate_cv = separate_cv,
    separate_cvbar = separate_cvbar,
    separate_alpha = separate_alpha,
    n_model_par_state = n_model_par_state,
    npar_state = npar_state,
    npar_total = npar_total,
    Dev = Dev,
    AIC = AIC,
    AICc = AICc,
    BIC = BIC
  )
}

# All 2^4 = 16 combinations of the command-line switches in ensemble.py.
models <- expand_grid(
  fit_sigma = c(FALSE, TRUE),
  separate_cv = c(FALSE, TRUE),
  separate_cvbar = c(FALSE, TRUE),
  separate_alpha = c(FALSE, TRUE)
) |>
  mutate(
    model = paste0(
      "ensemble_sigma", tf_label(fit_sigma),
      "_cv", tf_label(separate_cv),
      "_cvbar", tf_label(separate_cvbar),
      "_alpha", tf_label(separate_alpha)
    ),
    file = paste0(model, "_estimated.csv")
  )

dirF <- here("Result")

# Require all 16 fits so the resulting table is a genuine 16-model comparison.
missing_files <- models |>
  filter(!file.exists(file.path(dirF, file)))

if (nrow(missing_files) > 0) {
  stop(
    "The following ensemble result files are missing from ", dirF, ":\n",
    paste(missing_files$file, collapse = "\n")
  )
}

resl <- vector("list", nrow(models))

for (j in seq_len(nrow(models))) {
  spec <- models[j, ]
  tmp <- file.path(dirF, spec$file)
  res <- read.csv(tmp)

  # Keep ntrain == 0 when this column is present, as in the previous script.
  if ("ntrain" %in% names(res)) {
    res <- res[res$ntrain < 1, , drop = FALSE]
  }

  # Find the residual-sum-of-squares column.
  rss_col <- if ("RSE_1" %in% names(res)) {
    "RSE_1"
  } else if ("RSE" %in% names(res)) {
    "RSE"
  } else if ("RSS" %in% names(res)) {
    "RSS"
  } else {
    stop("None of RSE_1, RSE, or RSS found in ", spec$file)
  }

  # If several optimization seeds were saved for a state, retain the best fit
  # for that state before computing the model comparison statistic.
  if ("code" %in% names(res) && anyDuplicated(res$code)) {
    res <- res |>
      group_by(code) |>
      slice_min(order_by = .data[[rss_col]], n = 1, with_ties = FALSE) |>
      ungroup()
  }

  if (nrow(res) != nstates) {
    stop(
      spec$file, " contains ", nrow(res),
      " state-level fits after filtering; expected ", nstates, "."
    )
  }

  if (any(!is.finite(res[[rss_col]]) | res[[rss_col]] <= 0)) {
    stop("Non-positive or non-finite RSS/RSE found in ", spec$file)
  }

  # -2 log-likelihood up to constants common to all 16 models, with a
  # separately profiled residual variance for each state.
  Dev <- nsamples * sum(log(res[[rss_col]] / nsamples))

  resl[[j]] <- AIC_dete(
    mdl = spec$model,
    Dev = Dev,
    fit_sigma = spec$fit_sigma,
    separate_cv = spec$separate_cv,
    separate_cvbar = spec$separate_cvbar,
    separate_alpha = spec$separate_alpha
  )
}

ckh <- bind_rows(resl) |>
#  arrange(AIC) |>
  mutate(
    delta_AICc = AICc - min(AICc),
    akaike_weight = exp(-0.5 * delta_AICc) /
      sum(exp(-0.5 * delta_AICc))
  )

print(ckh, n = Inf)
write.csv(
  ckh,
  file.path(dirF, "complete_cmp_summary.csv"),
  row.names = FALSE
)
