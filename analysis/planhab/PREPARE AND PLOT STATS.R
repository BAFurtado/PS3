
# 1) PREPARE 


prepare_acps_runs <- function(
    data,
    acps,
    burn_in_months = 0, #meses de burn in para excluir da base tratada
    burn_in_date = NULL,
    scenario_col = "interest",
    acps_col = "processing_acps",
    date_col = "month",
    pol_cols = c("policy_mcmv", "policy_melhorias"),
    id_col = "X",
    start_date = "2010-02-01",
    end_date   = "2024-12-01",
    monthly_step = "month",
    scenario_order = c("baixa", "media", "alta"),
    strict_months = TRUE
) {
  
  resolve_id_col <- function(df, preferred = "X") {
    if (preferred %in% names(df)) return(preferred)
    if ("Unnamed: 0" %in% names(df)) return("Unnamed: 0")
    stop("Não encontrei coluna de índice 'X' nem 'Unnamed: 0'.")
  }
  
  normalize_policy01 <- function(v) {
    if (is.logical(v)) return(as.integer(v))
    if (is.numeric(v)) return(as.integer(v != 0))
    v <- as.character(v)
    v <- trimws(tolower(v))
    out <- rep(NA_integer_, length(v))
    out[v %in% c("true","t","1","yes","y")]  <- 1L
    out[v %in% c("false","f","0","no","n")] <- 0L
    out
  }
  
  pol_label01 <- function(mcmv, melh) {
    m1 <- normalize_policy01(mcmv)
    m2 <- normalize_policy01(melh)
    paste0("mcmv=", m1, ",melh=", m2)
  }
  
  order_scenarios <- function(levels_vec, desired_order) {
    levels_vec <- as.character(levels_vec)
    desired_order <- as.character(desired_order)
    in_order <- desired_order[desired_order %in% levels_vec]
    rest <- setdiff(levels_vec, desired_order)
    c(in_order, sort(rest))
  }
  
  # ---- validations ----
  stopifnot(is.data.frame(data))
  id_col <- resolve_id_col(data, preferred = id_col)
  
  needed <- c(acps_col, scenario_col, date_col, pol_cols, id_col)
  missing <- setdiff(needed, names(data))
  if (length(missing) > 0) stop(paste("Faltam colunas:", paste(missing, collapse = ", ")))
  
  data[[date_col]] <- as.Date(data[[date_col]])
  
  # ---- filter ACPS (prep scope) ----
  data <- data[data[[acps_col]] %in% acps, , drop = FALSE]
  if (nrow(data) == 0) stop("Nenhuma linha após filtrar ACPS.")
  
  # ---- expected months + burn-in (prep scope) ----
  expected_months <- seq(as.Date(start_date), as.Date(end_date), by = monthly_step)
  
  if (!is.null(burn_in_date)) {
    burn_in_date <- as.Date(burn_in_date)
    keep_months <- expected_months[expected_months >= burn_in_date]
  } else {
    burn_in_months <- as.integer(burn_in_months)
    if (burn_in_months < 0) stop("burn_in_months deve ser >= 0.")
    keep_months <- expected_months
    if (burn_in_months > 0) {
      if (burn_in_months >= length(expected_months)) stop("burn_in_months grande demais.")
      keep_months <- expected_months[(burn_in_months + 1):length(expected_months)]
    }
  }
  
  data <- data[data[[date_col]] %in% keep_months, , drop = FALSE]
  if (nrow(data) == 0) stop("Nenhuma linha após aplicar burn-in/datas.")
  
  # ---- create non-destructive policy label ----
  data$pol <- pol_label01(data[[pol_cols[1]]], data[[pol_cols[2]]])
  
  # ---- reconstruct run_id per (ACPS, scenario, pol) ----
  data$.__run_key <- interaction(
    data[[acps_col]],
    data[[scenario_col]],
    data$pol,
    drop = TRUE, lex.order = TRUE
  )
  data$run_id <- NA_integer_
  
  idx_by_key <- split(seq_len(nrow(data)), data$.__run_key)
  
  for (k in names(idx_by_key)) {
    idx <- idx_by_key[[k]]
    g <- data[idx, , drop = FALSE]
    
    ord <- order(g[[id_col]])
    idx_ord <- idx[ord]
    g_ord <- data[idx_ord, , drop = FALSE]
    
    months_g <- sort(unique(g_ord[[date_col]]))
    if (strict_months && !identical(months_g, keep_months)) {
      stop(paste0(
        "Grupo ", k, " não tem exatamente os meses esperados após burn-in. strict_months=TRUE."
      ))
    }
    keep_months_local <- if (strict_months) keep_months else months_g
    
    n_months <- length(keep_months_local)
    n_runs <- nrow(g_ord) / n_months
    if (abs(n_runs - round(n_runs)) > 1e-9) {
      stop(paste0(
        "Grupo ", k, " não fecha em múltiplo de meses. nrow=", nrow(g_ord),
        ", n_months=", n_months, ", n_runs=", n_runs
      ))
    }
    n_runs <- as.integer(round(n_runs))
    rid <- rep(seq_len(n_runs), each = n_months)
    
    if (strict_months) {
      for (r in seq_len(n_runs)) {
        ms <- g_ord[[date_col]][rid == r]
        ms <- ms[order(ms)]
        if (!identical(ms, keep_months_local)) {
          stop(paste0("Falha ao validar meses por run no grupo ", k, " (run_id=", r, ")."))
        }
      }
    }
    
    data$run_id[idx_ord] <- rid
  }
  
  scen_levels <- order_scenarios(unique(data[[scenario_col]]), scenario_order)
  
  attr(data, "keep_months") <- keep_months
  attr(data, "scenario_levels") <- scen_levels
  attr(data, "id_col_used") <- id_col
  
  data
}


## 2) PLOT 

plot_acps_policy_scenarios_prepared <- function(
    prepared_data,
    acps,
    vars,
    band_probs = c(0.25, 0.75),
    scenario_col = "interest",
    acps_col = "processing_acps",
    date_col = "month",
    pol_cols = c("policy_mcmv", "policy_melhorias"),
    
    # plot-time only
    policies_filter = NULL,  # NULL character vector ("mcmv=0,melh=0") list(c(0,0),...)
    
    smoothing = list(method = "none"),  # none | ma | lowess
    
    keep_scenario_mean_sd = TRUE, #plots mean and sd lines for ecah scenario/variable so that horizontal comparison becomes easier
    scenario_sd_multiplier = 1,
    scenario_lty_mean = 2,
    scenario_lty_sd = 3,
    
    show_policy_stats = TRUE, #plot média e sd para cada policy bundle
    policy_stats_method = "run_time_mean",  
    policy_stats_digits = 3, 
    policy_stats_pos = "topright",
    policy_stats_includes_n = TRUE,
    
    same_y_limits = TRUE, #normalizes y axis across scenarios for the same variable
    scenario_order = c("baixa", "media", "alta"),
    
    lwd_mean = 2,
    legend_pos = "bottomright",
    main_prefix = NULL
) {
  
  `%||%` <- function(a, b) if (!is.null(a)) a else b
  
  # --- policy helpers (must match preparation labels) ---
  normalize_policy01 <- function(v) {
    if (is.logical(v)) return(as.integer(v))
    if (is.numeric(v)) return(as.integer(v != 0))
    v <- as.character(v)
    v <- trimws(tolower(v))
    out <- rep(NA_integer_, length(v))
    out[v %in% c("true","t","1","yes","y")]  <- 1L
    out[v %in% c("false","f","0","no","n")] <- 0L
    out
  }
  
  pol_label01 <- function(mcmv, melh) {
    m1 <- normalize_policy01(mcmv)
    m2 <- normalize_policy01(melh)
    paste0("mcmv=", m1, ",melh=", m2)
  }
  
  pol_filter_to_labels <- function(policies_filter) {
    if (is.null(policies_filter)) return(NULL)
    if (is.character(policies_filter)) return(policies_filter)
    if (is.list(policies_filter)) {
      vapply(policies_filter, function(z) {
        if (length(z) != 2) stop("Cada elemento de policies_filter (list) deve ser c(mcmv,melh).")
        pol_label01(z[1], z[2])
      }, character(1))
    } else {
      stop("policies_filter deve ser NULL, character vector ou list de pares c(mcmv,melh).")
    }
  }
  
  # --- misc helpers ---
  order_scenarios <- function(levels_vec, desired_order) {
    levels_vec <- as.character(levels_vec)
    desired_order <- as.character(desired_order)
    in_order <- desired_order[desired_order %in% levels_vec]
    rest <- setdiff(levels_vec, desired_order)
    c(in_order, sort(rest))
  }
  
  fmt <- function(x, d) ifelse(is.finite(x), formatC(x, format = "f", digits = d), "NA")
  
  smooth_series <- function(x_dates, y, smoothing) {
    method <- tolower(smoothing$method %||% "none")
    if (method == "none") return(y)
    
    if (method == "ma") {
      k <- as.integer(smoothing$k %||% 7L)
      if (k <= 1) return(y)
      filt <- rep(1/k, k)
      ys <- as.numeric(stats::filter(y, filter = filt, sides = 2))
      na_idx <- which(!is.finite(ys))
      ys[na_idx] <- y[na_idx]
      return(ys)
    }
    
    if (method == "lowess") {
      f <- as.numeric(smoothing$f %||% 0.12)
      x_num <- as.numeric(x_dates)
      fit <- stats::lowess(x_num, y, f = f, iter = 0)
      ys <- approx(fit$x, fit$y, xout = x_num, rule = 2)$y
      return(ys)
    }
    
    stop("smoothing$method deve ser 'none', 'ma' ou 'lowess'.")
  }
  
  summarise_month <- function(v, probs) {
    v <- v[is.finite(v)]
    if (length(v) == 0) return(c(mean = NA_real_, lo = NA_real_, hi = NA_real_))
    qs <- as.numeric(quantile(v, probs = probs, na.rm = TRUE, type = 7))
    c(mean = mean(v, na.rm = TRUE), lo = qs[1], hi = qs[2])
  }
  
  build_monthly_summary <- function(df, varname) {
    key_df <- df[c(acps_col, scenario_col, pol_cols, date_col)]
    out <- aggregate(df[[varname]], by = key_df,
                     FUN = function(v) summarise_month(v, band_probs))
    mat <- out$x
    out$x <- NULL
    colnames(mat) <- c("mean", "lo", "hi")
    cbind(out, as.data.frame(mat))
  }
  
  # ---- checks ----
  stopifnot(is.data.frame(prepared_data))
  if (!("pol" %in% names(prepared_data))) stop("prepared_data precisa conter coluna 'pol'. Rode prepare_acps_runs().")
  if (!("run_id" %in% names(prepared_data))) stop("prepared_data precisa conter coluna 'run_id'. Rode prepare_acps_runs().")
  
  missing_vars <- setdiff(vars, names(prepared_data))
  if (length(missing_vars) > 0) stop(paste("Variáveis não encontradas:", paste(missing_vars, collapse = ", ")))
  
  keep_months <- attr(prepared_data, "keep_months")
  if (is.null(keep_months)) keep_months <- sort(unique(as.Date(prepared_data[[date_col]])))
  
  interests_all <- order_scenarios(unique(prepared_data[[scenario_col]]), scenario_order)
  
  keep_pol <- pol_filter_to_labels(policies_filter)
  
  for (ac in acps) {
    df_ac <- prepared_data[prepared_data[[acps_col]] == ac, , drop = FALSE]
    if (nrow(df_ac) == 0) next
    
    if (!is.null(keep_pol)) {
      df_ac <- df_ac[df_ac$pol %in% keep_pol, , drop = FALSE]
      if (nrow(df_ac) == 0) stop(paste0("Nenhuma linha para ACPS=", ac, " após aplicar policies_filter."))
    }
    
    nrow_plot <- length(vars)
    ncol_plot <- length(interests_all)
    
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar), add = TRUE)
    par(mfrow = c(nrow_plot, ncol_plot), mar = c(3.5, 3.8, 2.2, 1.0), oma = c(0, 0, 2.9, 0))
    
    for (v in vars) {
      summ <- build_monthly_summary(df_ac, v)
      summ[[date_col]] <- as.Date(summ[[date_col]])
      
      # IMPORTANT FIX: re-create policy label on the aggregated table
      summ$pol <- pol_label01(summ[[pol_cols[1]]], summ[[pol_cols[2]]])
      
      yr_global <- NULL
      if (same_y_limits) {
        yr_global <- range(summ$lo, summ$hi, finite = TRUE)
        if (!all(is.finite(yr_global))) yr_global <- NULL
      }
      
      for (intr in interests_all) {
        s_intr <- summ[summ[[scenario_col]] == intr, , drop = FALSE]
        if (nrow(s_intr) == 0) {
          plot.new()
          title(main = paste0("interest: ", intr))
          next
        }
        
        pols <- sort(unique(s_intr$pol))
        cols <- grDevices::rainbow(length(pols))
        
        xr <- range(keep_months, finite = TRUE)
        yr <- if (!is.null(yr_global)) yr_global else range(s_intr$lo, s_intr$hi, finite = TRUE)
        
        plot(
          xr, yr,
          type = "n",
          xlab = "",
          ylab = if (intr == interests_all[1]) v else "",
          main = paste0("interest: ", intr)
        )
        
        if (isTRUE(keep_scenario_mean_sd)) {
          ref_vals <- df_ac[df_ac[[scenario_col]] == intr, v]
          ref_vals <- ref_vals[is.finite(ref_vals)]
          if (length(ref_vals) > 1) {
            mu <- mean(ref_vals, na.rm = TRUE)
            sig <- stats::sd(ref_vals, na.rm = TRUE)
            abline(h = mu, lty = scenario_lty_mean, lwd = 1)
            if (is.finite(sig) && sig > 0) {
              abline(h = mu + scenario_sd_multiplier * sig, lty = scenario_lty_sd, lwd = 1)
              abline(h = mu - scenario_sd_multiplier * sig, lty = scenario_lty_sd, lwd = 1)
            }
          }
        }
        
        for (j in seq_along(pols)) {
          pj <- pols[j]
          dj <- s_intr[s_intr$pol == pj, , drop = FALSE]
          dj <- dj[order(dj[[date_col]]), , drop = FALSE]
          
          y_mean <- smooth_series(dj[[date_col]], dj$mean, smoothing)
          y_lo   <- smooth_series(dj[[date_col]], dj$lo, smoothing)
          y_hi   <- smooth_series(dj[[date_col]], dj$hi, smoothing)
          
          polygon(
            x = c(dj[[date_col]], rev(dj[[date_col]])),
            y = c(y_lo, rev(y_hi)),
            col = adjustcolor(cols[j], alpha.f = 0.20),
            border = NA
          )
          lines(dj[[date_col]], y_mean, col = cols[j], lwd = lwd_mean)
        }
        
        # Guard: only draw policy legend when it exists
        if (intr == interests_all[1] && length(pols) > 0) {
          legend(legend_pos, legend = pols, col = cols, lwd = lwd_mean, bty = "n", cex = 0.85)
        }
        
        if (isTRUE(show_policy_stats)) {
          df_panel <- df_ac[df_ac[[scenario_col]] == intr, , drop = FALSE]
          
          lbls <- character(0)
          tcols <- character(0)
          
          for (j in seq_along(pols)) {
            pj <- pols[j]
            dpol <- df_panel[df_panel$pol == pj, , drop = FALSE]
            if (nrow(dpol) == 0) next
            
            if (policy_stats_method == "run_time_mean") {
              run_means <- aggregate(dpol[[v]], by = list(run_id = dpol$run_id),
                                     FUN = function(z) mean(z, na.rm = TRUE))$x
              run_means <- run_means[is.finite(run_means)]
              mu <- if (length(run_means) > 0) mean(run_means) else NA_real_
              sig <- if (length(run_means) > 1) stats::sd(run_means) else NA_real_
              nrun <- length(run_means)
            } else stop("policy_stats_method deve ser 'run_time_mean' ou 'raw_points'.")
            
            #else if (policy_stats_method == "raw_points") {
            #  z <- dpol[[v]]
            #  z <- z[is.finite(z)]
            #  mu <- if (length(z) > 0) mean(z) else NA_real_
            #  sig <- if (length(z) > 1) stats::sd(z) else NA_real_
            #  nrun <- length(unique(dpol$run_id))
           # }
            
            lbl <- if (policy_stats_includes_n) {
              paste0("avg=", fmt(mu, policy_stats_digits),
                     " sd=", fmt(sig, policy_stats_digits),
                     " n=", nrun)
            } else {
              paste0("avg=", fmt(mu, policy_stats_digits),
                     " sd=", fmt(sig, policy_stats_digits))
            }
            
            lbls <- c(lbls, lbl)
            tcols <- c(tcols, cols[j])
          }
          
          if (length(lbls) > 0) {
            legend(policy_stats_pos, legend = lbls, text.col = tcols, bty = "n", cex = 0.9)
          }
        }
        
        if (v == vars[length(vars)]) {
          axis.Date(1, at = pretty(keep_months, n = 6), format = "%Y")
        }
      }
    }
    
    
    main_txt <- if (is.null(main_prefix)) {
      paste0("ACPS: ", ac,
             " | bands=", paste0(band_probs, collapse = "-"),
             " | smoothing=", tolower(smoothing$method %||% "none"),
             " | same_y_limits=", same_y_limits,
             " | scenario_mean±sd=", keep_scenario_mean_sd,
             " | policy_stats=", show_policy_stats)
    } else {
      paste0(main_prefix, " | ACPS: ", ac)
    }
    mtext(main_txt, outer = TRUE, cex = 1.0, line = 0.9)
  }
  
  invisible(prepared_data)
}


x <- read.csv("../../output/dados_sim_planhab/final_stats.csv", stringsAsFactors = FALSE, check.names = TRUE)


#como o arquivo original não tinha as identificações das runs distintas por acps, foi preciso inferir o index
#das runs e reconstruir cada uma para tirar estatísticas descritivas. essa função faz isso:

prep_x <- prepare_acps_runs(
  data = x,
  acps = unique(x$processing_acps), #quais acps incluir na base preparada
  burn_in_months = 36 #months to exclude (useful to analyze policy after simulation set-in)
)



# Now choose what to plot
plot_acps_policy_scenarios_prepared(
  prepared_data = prep_x,
  acps = c("ARACAJU"), #acps to plot, may plot multiple, use unique(x$processing_acps) to plot all
  vars = c("gini_index", "unemployment", "gdp_index"), #vars to plot
  policies_filter = c("mcmv=0,melh=0", "mcmv=1,melh=1"), #policy bundles to plot
  smoothing = list(method = "ma", k = 7), #methods: ma (use param k) or lowess (use param f. f usually around 2/3)
  policy_stats_pos = "topright"
)
